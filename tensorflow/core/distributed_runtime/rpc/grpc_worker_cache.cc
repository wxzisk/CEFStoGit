/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#define MYLOG VLOG(2)

namespace tensorflow {

namespace {

class GrpcWorkerCache : public WorkerCachePartial {
 public:
  // TODO(ncteisen): consider adding a config var or flag for this
  static constexpr const size_t kGrpcWorkerCacheThreadCount = 8;

  explicit GrpcWorkerCache(std::shared_ptr<GrpcChannelCache> channel_cache,
                           WorkerInterface* local_worker,
                           const string& local_target)
      : local_target_(local_target),
        local_worker_(local_worker),
        channel_cache_(channel_cache),
        threads_(kGrpcWorkerCacheThreadCount),
        next_round_robin_assignment_(0) {
    // NOTE: We don't yet have any reason to assign NUMA affinity to this
    // ThreadPool.  If there's only a single NIC it shouldn't make any
    // difference since presumably it is handling memory from all nodes.
    ThreadOptions options;
    options.numa_node = port::kNUMANoAffinity;
    const int kNumCallbackThreads = 10;
    callback_threadpool_.reset(new thread::ThreadPool(
        Env::Default(), options, "grpc_wcache_callback", kNumCallbackThreads,
        false /*low_latency_hint*/, nullptr /*allocator*/));
  }

  // Explicit destructor to control destruction order.
  ~GrpcWorkerCache() override {
    threads_.clear();  // Blocks until threads exit.
  }

  void ListWorkers(std::vector<string>* workers) const override {
    channel_cache_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    channel_cache_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* CreateWorker(const string& target) override {
    if (target == local_target_) {
      return local_worker_;
    } else {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (!channel) return nullptr;
      return NewGrpcRemoteWorker(
          channel, threads_[AssignWorkerToThread(target)].completion_queue(),
          callback_threadpool_.get(), &logger_);
    }
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
          << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }

  void SetLogging(bool v) override { logger_.SetLogging(v); }

  void ClearLogs() override { logger_.ClearLogs(); }

  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }

 private:
  // Thread wrapping class that drives work over a single gRPC
  // CompletionQueue.
  class GrpcWorkerCacheThread {
   public:
    GrpcWorkerCacheThread() {
      thread_.reset(Env::Default()->StartThread(
          ThreadOptions(), "grpc_worker_cache", [this]() {
            void* tag;
            bool ok;
            while (completion_queue_.Next(&tag, &ok)) {
              GrpcClientCQTag* callback_tag =
                  static_cast<GrpcClientCQTag*>(tag);
              callback_tag->OnCompleted(ok);
            }
          }));
    }

    ~GrpcWorkerCacheThread() {
      completion_queue_.Shutdown();
      thread_.reset();
    }

    ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

   private:
    ::grpc::CompletionQueue completion_queue_;
    std::unique_ptr<Thread> thread_;
  };  // GrpcWorkerCacheThread

  size_t AssignWorkerToThread(const string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  const string local_target_;
  WorkerInterface* const local_worker_;  // Not owned.
  std::shared_ptr<GrpcChannelCache> channel_cache_;
  WorkerCacheLogger logger_;
  std::vector<GrpcWorkerCacheThread> threads_;

  std::unique_ptr<thread::ThreadPool> callback_threadpool_;

  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);
};

}  // namespace

WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc) {
  return new GrpcWorkerCache(cc, nullptr, "");
}

WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, WorkerInterface* local_worker,
    const string& local_target) {
  return new GrpcWorkerCache(cc, local_worker, local_target);
}

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using VariableName = string;
using Priority = size_t;
using GradientName = string;

static std::unordered_map<string, uint64> device_incarnations;

static std::unordered_map<GradientName, Priority> priorities;

static ShapeHandle ShapeOrHandleShape(InferenceContext *c, int input) {
  auto *handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

static Status ApplyGradientDescentShapeFn(InferenceContext *c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                 // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused)); // alpha
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("SendGradient")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("gradient_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.
)doc");

REGISTER_OP("RecvApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("variable_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(ApplyGradientDescentShapeFn)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device and apply
GradientDescent algorithm to variable with the received tensor as delta.
)doc");

REGISTER_OP("SendParameter")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("variable_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.
)doc");

REGISTER_OP("RecvParameter")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device to recv_device.
)doc");

Status ReverseKey(const Rendezvous::ParsedKey &key,
                  Rendezvous::ParsedKey *reversed) {
  int64 device_incarnation = device_incarnations[string(key.dst_device)];
  string reversed_key_str = Rendezvous::CreateKey(
      string(key.dst_device), device_incarnation, string(key.src_device),
      string(key.edge_name), FrameAndIter(0, 0));

  return Rendezvous::ParseKey(reversed_key_str, reversed);
}

using DoneCallback = std::function<void(const Status &)>;

struct BaseOobUpdate {
  virtual ~BaseOobUpdate() {}
  virtual void Execute(DoneCallback done) = 0;
  virtual string Name() const = 0;
};

template <typename T> struct OobUpdate : public BaseOobUpdate {
  explicit OobUpdate(string variable_name, Rendezvous *rendezvous,
                     Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                     const Eigen::ThreadPoolDevice *device, Tensor var,
                     const Tensor &alpha)
      : variable_name_(variable_name), rendezvous_(rendezvous),
        parsed_key_(parsed_key), args_(args), device_(device),
        var_(std::move(var)), alpha_(alpha) {}

  ~OobUpdate() override {}

  void Execute(DoneCallback done) override {
    rendezvous_->RecvAsync(
        parsed_key_, args_,
        [this, done](const Status &s, const Rendezvous::Args &send_args,
                     const Rendezvous::Args &recv_args, const Tensor &delta,
                     bool is_dead) {
          if (!s.ok()) {
            return;
          }

          Rendezvous::ParsedKey ack_key;
          Status status = ReverseKey(parsed_key_, &ack_key);
          if (!status.ok()) {
            LOG(WARNING) << status;
          }
          rendezvous_->Send(ack_key, send_args, Tensor{}, false);
          MYLOG << "rendezvous send " << ack_key.edge_name;

          if (s.ok() && !is_dead) {
            MYLOG << "Start gradient update to " << variable_name_;
            typename TTypes<T>::Flat var = var_.flat<T>();
            typename TTypes<T>::ConstFlat grad = delta.flat<T>();
            typename TTypes<T>::ConstScalar lr = alpha_.scalar<T>();
            var.device(*device_) -= grad * lr();
            MYLOG << "Finish gradient update to " << variable_name_;
          }
          done(s);
        });
  }

  string Name() const override { return variable_name_; }

  string variable_name_;

  Rendezvous *rendezvous_;
  Rendezvous::ParsedKey parsed_key_;
  Rendezvous::Args args_;

  const Eigen::ThreadPoolDevice *device_;
  Tensor var_;
  const Tensor alpha_;
};

struct GradientPush {

  explicit GradientPush(string gradient_name, Rendezvous *rendezvous,
                        Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                        const Tensor &gradient, bool is_dead)
      : gradient_name_(gradient_name), rendezvous_(rendezvous),
        parsed_key_(parsed_key), args_(args), gradient_(gradient),
        is_dead_(is_dead) {}

  void Execute(DoneCallback done) {
    rendezvous_->Send(parsed_key_, args_, gradient_, is_dead_);

    Rendezvous::ParsedKey ack_key;
    Status status = ReverseKey(parsed_key_, &ack_key);
    if (!status.ok()) {
      LOG(WARNING) << status;
    }

    int64 start = Env::Default()->NowMicros();
    rendezvous_->RecvAsync(
        ack_key, args_,
        [this, done, start](const Status &s, const Rendezvous::Args &send_args,
                            const Rendezvous::Args &recv_args, const Tensor &t,
                            bool is_dead) {
          if (!s.ok()) {
            LOG(WARNING) << s;
          } else {
            int64 duration = Env::Default()->NowMicros() - start;
            MYLOG << "Ack RTT for " << gradient_name_ << " takes " << duration
                    << " us";
          }
          done(s);
        });
  }

  size_t NumBytes() const { return gradient_.TotalBytes(); }

  size_t Priority() const { return priorities[gradient_name_]; }

  string gradient_name_;

  Rendezvous *rendezvous_;
  Rendezvous::ParsedKey parsed_key_;
  Rendezvous::Args args_;
  const Tensor gradient_;
  bool is_dead_;
};

class OobUpdateManager {
public:
  explicit OobUpdateManager() : bytes_in_flight_(0) {}

  void Schedule(string gradient_name, Rendezvous *rendezvous,
                Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                const Tensor &gradient, bool is_dead) {
    GradientPush *push = new GradientPush(gradient_name, rendezvous, parsed_key,
                                          args, gradient, is_dead);
    Schedule(push);
  }

  void Schedule(GradientPush *push) {
    MYLOG << "Scheduling gradient " << push->gradient_name_;
    push->Execute([this, push](const Status &s) {
      MYLOG << "Finished pushing gradient " << push->gradient_name_;
      delete push;
    });
  }

  template <typename T>
  void RecvUpdate(string variable_name, Rendezvous *rendezvous,
                  Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                  const Eigen::ThreadPoolDevice *device, Tensor var,
                  const Tensor &alpha) {
    string src_device = string(parsed_key.src_device);
    MYLOG << "Fetching updates to " << variable_name;
    BaseOobUpdate *update = new OobUpdate<T>(
        variable_name, rendezvous, parsed_key, args, device, var, alpha);
    update->Execute([this, update, src_device](const Status &s) {
      Ready(src_device, update->Name(), s);
      delete update;
    });
  }

  void Ready(string device, string variable_name, Status s) {
    DoneCallback done;
    string node, unused;
    DeviceNameUtils::SplitDeviceName(device, &node, &unused);
    string key = strings::StrCat(node, variable_name);
    //string key = strings::StrCat(device, variable_name);
    {
      mutex_lock l(mu_);
      auto iter = callbacks_.find(key);
      if (iter != std::end(callbacks_)) {
        done = std::move(iter->second);
        callbacks_.erase(iter);
      } else {
        decltype(completed_status_)::iterator _;
        bool success;
        std::tie(_, success) = completed_status_.insert(std::make_pair(key, s));
      }
    }
    if (done) {
      done(s);
    }
  }

  void Poll(string device, string variable_name, DoneCallback done) {
    Status s = Status::OK();
    bool valid = false;
    string node, unused;
    DeviceNameUtils::SplitDeviceName(device, &node, &unused);
    string key = strings::StrCat(node, variable_name);
    //string key = strings::StrCat(device, variable_name);
    {
      mutex_lock l(mu_);
      if (seen_keys_.find(key) == std::end(seen_keys_)) {
        seen_keys_.insert(key);
        valid = true;
      }
    }
    if (!valid) {
      mutex_lock l(mu_);
      auto iter = completed_status_.find(key);
      if (iter != std::end(completed_status_)) {
        s = iter->second;
        valid = true;
        completed_status_.erase(iter);
      } else {
        decltype(callbacks_)::iterator _;
        bool success;
        std::tie(_, success) =
            callbacks_.insert(std::make_pair(key, std::move(done)));
      }
    }
    if (valid) {
      done(s);
    }
  }

  static OobUpdateManager *Get() {
    static OobUpdateManager *manager = new OobUpdateManager;
    return manager;
  }

private:
  struct Comparator {
    bool operator()(GradientPush *a, GradientPush *b) const {
      return a->Priority() > b->Priority();
    }
  };

  std::atomic<size_t> bytes_in_flight_;

  mutex mu_;
  // PS side
  std::unordered_map<string, DoneCallback> callbacks_ GUARDED_BY(mu_);
  std::unordered_map<string, Status> completed_status_ GUARDED_BY(mu_);
  std::set<string> seen_keys_ GUARDED_BY(mu_);
};

class SendGradientOp : public OpKernel {
public:
  explicit SendGradientOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gradient_name", &gradient_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
  }

  void Compute(OpKernelContext *ctx) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->Schedule(gradient_name_, ctx->rendezvous(),
                                      parsed_key, args, ctx->input(0),
                                      ctx->is_input_dead());
  }

private:
  string gradient_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendGradientOp);
};

template <typename T> class RecvApplyGradientDescentOp : public AsyncOpKernel {
public:
  explicit RecvApplyGradientDescentOp(OpKernelConstruction *ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variable_name", &variable_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {

    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    Rendezvous::Args args;
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_nic_compatible(true);
    alloc_attrs.set_on_host(true);
    args.alloc_attrs = alloc_attrs;
    args.device_context = ctx->op_device_context();

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->RecvUpdate<T>(
        variable_name_, ctx->rendezvous(), parsed_key, args,
        &ctx->eigen_cpu_device(), ctx->mutable_input(0, false), ctx->input(1));

    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->SetStatus(Status::OK());
    done();
  }

private:
  string variable_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvApplyGradientDescentOp);
};

class SendParameterOp : public AsyncOpKernel {
public:
  explicit SendParameterOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variable_name", &variable_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
    device_incarnations.insert(
        std::make_pair(send_device_, send_device_incarnation_));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->Poll(
        recv_device_, variable_name_,
        [this, ctx, parsed_key, args, done](const Status &s) {
          if (!s.ok()) {
            LOG(WARNING) << s;
            ctx->SetStatus(s);
            done();
            return;
          }
          ctx->rendezvous()->Send(parsed_key, args, ctx->input(0),
                                  ctx->is_input_dead());
          done();
        });
  }

private:
  string variable_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendParameterOp);
};

class RecvParameterOp : public AsyncOpKernel {
public:
  explicit RecvParameterOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
    device_incarnations.insert(
        std::make_pair(send_device_, send_device_incarnation_));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    ctx->rendezvous()->RecvAsync(parsed_key, args,
                                 [ctx, done](const Status &s,
                                             const Rendezvous::Args &send_args,
                                             const Rendezvous::Args &recv_args,
                                             const Tensor &t, bool is_dead) {
                                   ctx->SetStatus(s);
                                   if (s.ok() && !is_dead) {
                                     ctx->set_output(0, t);
                                   }
                                   done();
                                 });
  }

private:
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvParameterOp);
};

REGISTER_KERNEL_BUILDER(Name("SendGradient").Device(DEVICE_CPU),
                        SendGradientOp);
REGISTER_KERNEL_BUILDER(Name("SendGradient").Device(DEVICE_GPU),
                        SendGradientOp);
REGISTER_KERNEL_BUILDER(Name("RecvApplyGradientDescent")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        RecvApplyGradientDescentOp<float>);
REGISTER_KERNEL_BUILDER(Name("SendParameter").Device(DEVICE_CPU),
                        SendParameterOp);
REGISTER_KERNEL_BUILDER(Name("RecvParameter").Device(DEVICE_CPU),
                        RecvParameterOp);
REGISTER_KERNEL_BUILDER(Name("RecvParameter").Device(DEVICE_GPU),
                        RecvParameterOp);

struct WorkerRewriteTask {
  Node *send_op;
  Node *grad_op;
  int grad_index;
  std::vector<NodeBuilder::NodeOut> send_out_nodes;
};

class WorkerRewritePass : public GraphOptimizationPass {
public:
  Status Run(const GraphOptimizationPassOptions &options) override {

    VLOG(1) << "Successfully loaded WorkerRewritePass";
    std::unordered_map<string, std::unique_ptr<Graph>> *partition_graphs =
        options.partition_graphs;
    if (partition_graphs == nullptr) {
      return errors::Internal("Partitioned graphs is not available");
    }

    for (auto &kv : *partition_graphs) {
      if (str_util::StrContains(kv.first, "worker")) {
        Graph *graph = kv.second.get();

        std::vector<WorkerRewriteTask> tasks;

        for (Node *node : graph->op_nodes()) {
          if (node->IsSend()) {
            Node *send = node;
            Node *grad;
            TF_RETURN_IF_ERROR(send->input_node(0, &grad));
            auto iter = priorities.find(grad->name());
            if (iter != std::end(priorities)) {
              WorkerRewriteTask task;
              task.send_op = send;
              task.grad_op = grad;
              tasks.push_back(task);
            }
          }
        }

        for (Edge *edge : graph->edges()) {
          for (auto &task : tasks) {
            if (edge->src() == task.send_op) {
              task.send_out_nodes.emplace_back(edge->dst(), edge->dst_input());
            } else if (edge->src() == task.grad_op &&
                       edge->dst() == task.send_op) {
              task.grad_index = edge->src_output();
            }
          }
        }

        for (WorkerRewriteTask &task : tasks) {
          DataType dtype;
          TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "T", &dtype));
          string tensor_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "tensor_name", &tensor_name));
          string send_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "send_device", &send_device));
          string recv_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "recv_device", &recv_device));
          int64 send_device_incarnation;
          TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(),
                                         "send_device_incarnation",
                                         &send_device_incarnation));

          NodeBuilder builder(task.send_op->name(), "SendGradient");
          builder.Input(task.grad_op, task.grad_index);
          builder.Attr("T", dtype);
          builder.Attr("gradient_name", task.grad_op->name());
          builder.Attr("tensor_name", tensor_name);
          builder.Attr("send_device", send_device);
          builder.Attr("recv_device", recv_device);
          builder.Attr("send_device_incarnation", send_device_incarnation);

          Node *node;
          TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

          graph->RemoveNode(task.send_op);

          for (const auto &out_node : task.send_out_nodes) {
            if (out_node.index == Graph::kControlSlot) {
              graph->AddControlEdge(node, out_node.node);
            } else {
              graph->AddEdge(node, 0, out_node.node, out_node.index);
            }
          }
          TF_RETURN_IF_ERROR(graph->IsValidNode(node));

          LOG(INFO) << "Replacing gradient " << task.grad_op->name();
        }

        std::unordered_map<Node *, std::vector<NodeBuilder::NodeOut>> recv_ops;

        for (Node *node : graph->nodes()) {
          if (node->IsRecv()) {
            recv_ops.insert(
                std::make_pair(node, std::vector<NodeBuilder::NodeOut>()));
          }
        }

        for (Edge *edge : graph->edges()) {
          if (edge->src()->IsRecv()) {
            Node *recv = edge->src();
            auto iter = recv_ops.find(recv);
            if (iter != std::end(recv_ops)) {
              iter->second.emplace_back(edge->dst(), edge->dst_input());
            }
          }
        }

        for (auto &p : recv_ops) {
          Node *recv_op = p.first;
          std::vector<NodeBuilder::NodeOut> &out_nodes = p.second;

          DataType dtype;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "tensor_type", &dtype));
          string tensor_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "tensor_name", &tensor_name));
          string send_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "send_device", &send_device));
          string recv_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "recv_device", &recv_device));
          int64 send_device_incarnation;
          TF_RETURN_IF_ERROR(GetNodeAttr(recv_op->attrs(),
                                         "send_device_incarnation",
                                         &send_device_incarnation));

          NodeBuilder builder(recv_op->name(), "RecvParameter");
          builder.Attr("tensor_type", dtype);
          builder.Attr("tensor_name", tensor_name);
          builder.Attr("send_device", send_device);
          builder.Attr("recv_device", recv_device);
          builder.Attr("send_device_incarnation", send_device_incarnation);

          Node *node;
          TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

          graph->RemoveNode(recv_op);

          for (const auto &out_node : out_nodes) {
            if (out_node.index == Graph::kControlSlot) {
              graph->AddControlEdge(node, out_node.node);
            } else {
              graph->AddEdge(node, 0, out_node.node, out_node.index);
            }
          }
          TF_RETURN_IF_ERROR(graph->IsValidNode(node));
        }
      }
    }

    return Status::OK();
  }
};

struct PSRewriteTask {
  VariableName variable_name;
  Node *update_op;
  Node *recv_op;
  Node *var_op;
  Node *send_op;
  std::vector<NodeBuilder::NodeOut> update_out_nodes;
  std::vector<NodeBuilder::NodeOut> send_out_nodes;
};

class PSRewritePass : public GraphOptimizationPass {
public:
  Status Run(const GraphOptimizationPassOptions &options) override {

    VLOG(1) << "Successfully loaded PSRewritePass";
    std::unordered_map<string, std::unique_ptr<Graph>> *partition_graphs =
        options.partition_graphs;
    if (partition_graphs == nullptr) {
      return errors::Internal("Partitioned graphs is not available");
    }

    std::unordered_map<string, PSRewriteTask> task_map;

    for (auto &kv : *partition_graphs) {
      if (str_util::StrContains(kv.first, "ps")) {
        Graph *graph = kv.second.get();
        for (Node *node : graph->op_nodes()) {
          if (node->type_string() == "ApplyGradientDescent") {
            Node *var, *grad;
            Status s = node->input_node(0, &var);
            if (!s.ok() || !IsVariable(var)) {
              return errors::Internal("Cannot find variable for apply");
            }
            s = node->input_node(2, &grad);
            if (!s.ok() || !IsRecv(grad)) {
              return errors::Internal("Cannot find grad for apply");
            }

            PSRewriteTask task = {};
            task.variable_name = var->name();
            task.update_op = node;
            task.recv_op = grad;
            task_map.insert(std::make_pair(var->name(), task));
          }
        }

        for (Node *node : graph->op_nodes()) {
          if (IsSend(node)) {
            Node *var;
            TF_RETURN_IF_ERROR(node->input_node(0, &var));
            auto iter = task_map.find(var->name());
            if (iter != std::end(task_map)) {
              iter->second.var_op = var;
              iter->second.send_op = node;
            }
          }
        }

        for (Edge *edge : graph->edges()) {
          for (auto &kv : task_map) {
            if (edge->src() == kv.second.update_op) {
              kv.second.update_out_nodes.emplace_back(edge->dst(),
                                                      edge->dst_input());
            } else if (edge->src() == kv.second.send_op) {
              kv.second.send_out_nodes.emplace_back(edge->dst(),
                                                    edge->dst_input());
            }
          }
        }

        for (auto &kv : task_map) {
          PSRewriteTask &task = kv.second;
          {
            DataType dtype;
            TF_RETURN_IF_ERROR(
                GetNodeAttr(task.update_op->attrs(), "T", &dtype));
            string tensor_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "tensor_name",
                                           &tensor_name));
            string send_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "send_device",
                                           &send_device));
            string recv_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "recv_device",
                                           &recv_device));
            int64 send_device_incarnation;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(),
                                           "send_device_incarnation",
                                           &send_device_incarnation));

            NodeBuilder builder(task.update_op->name(),
                                "RecvApplyGradientDescent");
            builder.Attr("T", dtype);
            builder.Attr("variable_name", task.variable_name);
            builder.Attr("tensor_name", tensor_name);
            builder.Attr("send_device", send_device);
            builder.Attr("recv_device", recv_device);
            builder.Attr("send_device_incarnation", send_device_incarnation);

            Node *var, *lr;
            TF_RETURN_IF_ERROR(task.update_op->input_node(0, &var));
            builder.Input(var, 0);
            TF_RETURN_IF_ERROR(task.update_op->input_node(1, &lr));
            builder.Input(lr, 0);

            Node *fused_op;
            TF_RETURN_IF_ERROR(builder.Finalize(graph, &fused_op));

            graph->RemoveNode(task.recv_op);
            graph->RemoveNode(task.update_op);

            for (const auto &out_node : task.update_out_nodes) {
              if (out_node.index == Graph::kControlSlot) {
                graph->AddControlEdge(fused_op, out_node.node);
              } else {
                graph->AddEdge(fused_op, 0, out_node.node, out_node.index);
              }
            }
            TF_RETURN_IF_ERROR(graph->IsValidNode(fused_op));
          }
          {
            DataType dtype;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "T", &dtype));
            string tensor_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "tensor_name",
                                           &tensor_name));
            string send_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "send_device",
                                           &send_device));
            string recv_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "recv_device",
                                           &recv_device));
            int64 send_device_incarnation;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(),
                                           "send_device_incarnation",
                                           &send_device_incarnation));

            Node *node;
            NodeBuilder builder(task.send_op->name(), "SendParameter");
            builder.Input(task.var_op);
            builder.Attr("T", dtype);
            builder.Attr("variable_name", task.variable_name);
            builder.Attr("tensor_name", tensor_name);
            builder.Attr("send_device", send_device);
            builder.Attr("recv_device", recv_device);
            builder.Attr("send_device_incarnation", send_device_incarnation);

            TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

            graph->RemoveNode(task.send_op);

            for (const auto &out_node : task.send_out_nodes) {
              if (out_node.index == Graph::kControlSlot) {
                graph->AddControlEdge(node, out_node.node);
              } else {
                graph->AddEdge(node, 0, out_node.node, out_node.index);
              }
            }
            TF_RETURN_IF_ERROR(graph->IsValidNode(node));
          }
          LOG(INFO) << "Replacing variable " << task.variable_name;
        }
      }
    }

    return Status::OK();
  }
};

// for Geryon
mutex allocating_mu;
bool finish_allocating_ = false GUARDED_BY(allocating_mu);
bool is_worker_ = false;
bool is_chief_ = false;
int64 last_step_id_ = 0;
int64 steps_ = 0;
struct timespec current_tv, last_tv;
int64 iteration_time_;
std::unordered_map<int, int64> ps_step_id_pool_;
std::unordered_map<int, int64> worker_step_id_pool_;
std::map<std::pair<string, int>, string> ip_table_;
std::map<string, int> socket_table_;
uint32_t mapping_mode;
uint8_t flow_priority;
std::vector<uint8_t> priority_vec;

TensorOrder model_parameter_order_;
TensorOrder parameter_order_;
TensorOrder gradient_order_;

PriorityTable parameter_table_;
PriorityTable gradient_table_;
std::unordered_map<size_t, PriorityTable> parameter_table_map_;
std::unordered_map<size_t, PriorityTable> gradient_table_map_;
pthread_rwlock_t gradient_table_rwlock_;

struct timespec ccp_start_tv_;
bool should_profile_ = false;
std::unordered_map<string, bool> edges_set_;
std::unordered_map<string, std::vector<size_t>> worker_power_vector_;
std::unordered_map<string, size_t> worker_power_;
std::unordered_map<size_t, std::vector<size_t>> powergrade_to_worker_;
std::unordered_map<size_t, size_t> worker_to_powergrade_;
string last_egde_name;

// bayesian optmization module
int para_map_sockfd_ = -1;
int num_heter_workers_ = 1;
std::unordered_map<string, double> probe_point_;

class GraphRewritePass : public GraphOptimizationPass {
public:
  struct VariableInfo {
    Node *grad_op;
    Node *apply_op;
    string variable_name;
  };

  void GetVariableOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator = {},
                         const EdgeFilter& edge_filter = {}) {
    order->clear();
    for (Node *node : g.op_nodes()) {
      if (node->type_string() == "SparseSoftmaxCrossEntropyWithLogits" ||
          node->type_string() == "Mean") {
        ReverseBFSFrom(g, {node}, nullptr, [order](Node* n) { order->push_back(n); });
        break;
      }
    }
    reverse(order->begin(),order->end());
  }

  Status Run(const GraphOptimizationPassOptions &options) override {
    VLOG(INFO) << "Successfully loaded GraphRewritePass";   
    Graph *graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal("Graph is not available");
    }

    std::unordered_map<string, VariableInfo> variables;

    for (Node *node : graph->op_nodes()) {
      if (node->type_string() == "ApplyGradientDescent") {
        Node *var, *grad;
        Status s = node->input_node(0, &var);
        if (!s.ok() || !IsVariable(var)) {
          return errors::Internal("Cannot find variable for apply");
        }
        s = node->input_node(2, &grad);
        if (!s.ok()) {
          return errors::Internal("Cannot find gradient for apply");
        }

        VariableInfo info = {};
        info.variable_name = var->name();
        info.apply_op = node;
        info.grad_op = grad;
        variables.insert(std::make_pair(var->name(), info));
      }
    }

    std::vector<Node *> order;
    //GetReversePostOrder(*graph, &order);
    GetVariableOrder(*graph, &order);

    for (Node *node : order) {
      if (node->IsVariable()) {
        auto iter = variables.find(node->name());
        if (iter != std::end(variables)) {
          string var = node->name();
          parameter_order_.push_back(var);
          string grad = iter->second.grad_op->name();
          gradient_order_.push_back(grad);
          priorities.insert(std::make_pair(grad, 0)); // priorityies
          string::size_type idx = var.find("/part_");
          if (idx != string::npos)
            var = var.substr(0, idx);

          TensorOrder::iterator it = find(model_parameter_order_.begin(), model_parameter_order_.end(), var);
          if (it == model_parameter_order_.end()) {
            model_parameter_order_.push_back(var);
            LOG(INFO) << "Sorting variable " << var;
          }
        }
      }
    }

    return Status::OK();
  }
};

/* REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      WorkerRewritePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      PSRewritePass); */
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      GraphRewritePass);

}  // namespace tensorflow
