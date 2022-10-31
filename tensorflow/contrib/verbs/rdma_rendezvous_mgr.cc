/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/rdma_rendezvous_mgr.h"
#include <unordered_set>
#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class RdmaRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RdmaRemoteRendezvous(const WorkerEnv* env, int64 step_id, RdmaMgr* rdma_mgr)
      : BaseRemoteRendezvous(env, step_id), rdma_mgr_(rdma_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RdmaRemoteRendezvous() override {}
  RdmaMgr* rdma_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRemoteRendezvous);
};

void RdmaRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  Status s;
  // parse src_name and dst_name
  string src_name, dst_name, src_device, dst_device;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                        &src_device) ||
      !DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &dst_device)) {
    s = errors::Internal("Could not parse src or dst name.");
  }
  if (!s.ok()) {
    LOG(ERROR) << "s is not ok, error code " << s.error_message();
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);

  RdmaChannel* rc;
  if (flow_priority) {
    size_t priority;
    {
      mutex_lock ml(allocating_mu);
      if (!finish_allocating_ && 
            is_new_iteration(is_worker_, step_id_, last_step_id_, src_name, std::string(parsed.edge_name))) { // a new step
        last_step_id_ = step_id_;
        steps_ = steps_ + 1;
        //VLOG(INFO) << "New iteration." << steps_;
      
        if (steps_ >= 1 && steps_ < 8) {
          steps_sync(is_worker_, is_chief_);
        }
        else if (steps_ == 8) {
          finish_allocating_ = allocating_and_distribution(steps_, -2, is_worker_, is_chief_);
        }
        else if (steps_ > 9 && steps_ < 13) { // profile information about each worker's computation power
          if (!is_worker_ && is_chief_) {
            should_profile_ = true;
            clock_gettime(CLOCK_REALTIME, &ccp_start_tv_);
          }
        }
        else if (steps_ == 13) { // send information about each worker's computation power to worker 0
          should_profile_ = false;
          worker_power_sync(is_worker_, is_chief_);
          num_heter_workers_ = cluster_worker_power(is_worker_, is_chief_, 
                                worker_power_, (double)set_param(10, "PERF_GAP")/100);  //default performace gap = 10%
          VLOG(INFO) << "   worker  iter_time";
          for (auto w : worker_power_)
            VLOG(INFO) << "\t" << w.first << "\t" << w.second;
          VLOG(INFO) << "   worker  powergrade";
          for (auto p : worker_to_powergrade_)
            VLOG(INFO) << "\t" << p.first << "\t" << p.second;
          /* VLOG(INFO) << "powergrade_to_worker_:" ;
          for (auto p : powergrade_to_worker_) {
            VLOG(INFO) << "grade = " << p.first;
            for (auto w: p.second)
              VLOG(INFO) << w;
          } */
          finish_allocating_ = allocating_and_distribution(steps_, -1, is_worker_, is_chief_);
        }
        struct timespec tmp_tv;
        clock_gettime(CLOCK_REALTIME, &tmp_tv);
        if (steps_ % 5 == 0) {
          clock_gettime(CLOCK_REALTIME, &current_tv);
          if (steps_ >= 25) {
            iteration_time_ = (current_tv.tv_sec - last_tv.tv_sec) * 1e3 + (current_tv.tv_nsec - last_tv.tv_nsec) / 1e6;
            VLOG(INFO) << "iteration time : " << iteration_time_;

            finish_allocating_ = allocating_and_distribution(steps_, iteration_time_, is_worker_, is_chief_);
          }

          clock_gettime(CLOCK_REALTIME, &last_tv);
        }
      }
    }

    priority = FindPriority(std::string(parsed.edge_name), is_worker_, src_name);
    if (find(priority_vec.begin(), priority_vec.end(), priority) == priority_vec.end()) {
      priority = priority_vec.back();
    }
    //VLOG(INFO) << priority << "\t" << std::string(parsed.edge_name);
    rc = rdma_mgr_->FindChannel(src_name + "+" + std::to_string(priority));
  }
  else
    rc = rdma_mgr_->FindChannel(src_name);
  
  string key(parsed.FullKey());
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);

  Device* dst_dev;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  RdmaTensorRequest* request =
      rc->InsertTensorRequest(key, step_id_, dst_dev, recv_args, done);
  request->Start();
}

RdmaRendezvousMgr::RdmaRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RdmaRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env) {
  return new RdmaRemoteRendezvous(worker_env, step_id, rdma_mgr_);
}

}  // end namespace tensorflow

#endif
