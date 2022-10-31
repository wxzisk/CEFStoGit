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

#include "tensorflow/contrib/verbs/rdma_mgr.h"
#include <fstream>
#include <vector>
#include "tensorflow/contrib/verbs/grpc_verbs_client.h"
#include "tensorflow/contrib/verbs/verbs_service.pb.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

RdmaMgr::RdmaMgr(const WorkerEnv* const worker_env,
                 GrpcChannelCache* const channel_cache)
    : worker_env_(worker_env), channel_cache_(channel_cache) {
  rdma_adapter_ = new RdmaAdapter(worker_env_);
  // hardcoded to default session (legacy_session_)
  // TODO: use WorkerSessionForSession
  // need to pass in session handle
  local_worker_ = worker_env_->session_mgr->LegacySession()->worker_name;
  std::vector<string> workers;
  worker_env_->session_mgr->LegacySession()->worker_cache->ListWorkers(
      &workers);
  num_remote_workers_ = workers.size() - 1;
  VLOG(2) << "rmda_mgr on local worker: " << local_worker_;

  if (flow_priority) {
    pthread_rwlock_init(&gradient_table_rwlock_, NULL);
    if (str_util::StrContains(local_worker_, "/job:worker"))
      is_worker_ = true;
    if (str_util::StrContains(local_worker_, "/task:0"))
      is_chief_ = true;

    if (is_worker_ && is_chief_) {  // worker 0: server
      struct sockaddr_in local_socket, remote_socket; 
      socklen_t len;
      int s_socket, socket_conn;
      s_socket = socket(AF_INET, SOCK_STREAM, 0);
      if(s_socket < 0)
        VLOG(INFO) << "Faile to create socket!";
      local_socket.sin_family = AF_INET;
      local_socket.sin_port = htons(set_param(5006, "PM_PORT"));
      local_socket.sin_addr.s_addr = htonl(INADDR_ANY);

      bind(s_socket, (struct sockaddr *)&local_socket, sizeof(local_socket));  
      listen(s_socket, 1024);
      VLOG(INFO) << "Start to listen ...";

      mapping_mode = set_param(0, "MAPPING_MODE");
      switch (mapping_mode) {
        case 0:
          VLOG(INFO) << "MAPPING_MODE: layer_function_allocating";
          break;
        case 1:
          VLOG(INFO) << "MAPPING_MODE: equal_allocating";
          break;
        case 2:
          VLOG(INFO) << "MAPPING_MODE: specify_allocating";
          break;
        case 3:
          VLOG(INFO) << "MAPPING_MODE: bo_allocating";
          break;
        default:
          CHECK(0) << "MAPPING_MODE only allows 0 ~ 3 by now.";
          break;
      }
      bool need_pm_module = mapping_mode == 3 ? true : false;
      while(socket_table_.size() < num_remote_workers_ || 
            (para_map_sockfd_ == -1 && need_pm_module)) {
        len = sizeof(struct sockaddr_in);
        socket_conn = accept(s_socket, (struct sockaddr *)&remote_socket, &len);

        if(socket_conn >= 0){  
          char recv_buf[32];
          memset(recv_buf, 0, 32);
          recv(socket_conn, recv_buf, 32, 0);
          VLOG(INFO) << "Socket connected to " << string(recv_buf) << " with fd " << socket_conn << ".";
          if (string(recv_buf) == string("ParameterMapping"))
            para_map_sockfd_ = socket_conn;
          else
            socket_table_.insert(std::make_pair(string(recv_buf), socket_conn));
        }
      }      
    }
    else{ // all pses and other workers: client
      int c_socket, conn;
      struct sockaddr_in remote_socket;
      c_socket = socket(AF_INET, SOCK_STREAM, 0);
      remote_socket.sin_family = AF_INET;
      remote_socket.sin_port = htons(set_param(5006, "PM_PORT"));
      VLOG(INFO) << "The master's IP is " << ip_table_[std::make_pair("worker", 0)];
      remote_socket.sin_addr.s_addr = inet_addr(ip_table_[std::make_pair("worker", 0)].c_str());
      while(1) {
        conn = connect(c_socket, (struct sockaddr *)&remote_socket, sizeof(struct sockaddr_in));
        if(conn >= 0){
          char send_buf[32];
          DeviceNameUtils::ParsedName pn;
          DeviceNameUtils::ParseFullName(local_worker_, &pn);
          strcpy(send_buf, strings::StrCat(pn.job, std::to_string(pn.task)).c_str());
          send(c_socket, send_buf, 32, 0);
          VLOG(INFO) << "Socket connected to worker0 with fd " << c_socket << ".";
          socket_table_.insert(std::make_pair("worker0", c_socket));
          break;
        }
        else {
          VLOG(INFO) << "Fail to connect to worker0. Retrying...";
          sleep(2);
        }     
      }
    }
  }

  for (size_t i = 0; i < workers.size(); i++) {
    if (local_worker_.compare(workers[i]) != 0) {
      if (!flow_priority){
        channel_table_.insert({workers[i],
             new RdmaChannel(rdma_adapter_, local_worker_, workers[i])});
      }
      else {
        // each priority has a single rdma flow
        std::vector<uint8_t>::iterator it;
        for(it = priority_vec.begin(); it != priority_vec.end(); it++)
          channel_table_.insert({workers[i] + "+" + std::to_string(*it),
               new RdmaChannel(rdma_adapter_, local_worker_, workers[i])});
      } 
    }
  }
}

// Setup Rdma channels between peers.
// This is done at the beginning of the server setup.

void RdmaMgr::SetupChannels() {
  string worker_name;
  string worker_name_with_prio;
  string priority;
  for (const auto& p : channel_table_) {
    string worker_name = p.first;
    RDMA_LOG(2) << "Connecting to remote node " << worker_name;
    if (flow_priority) {
      int pos = p.first.find('+');
      worker_name_with_prio = p.first;
      worker_name = worker_name_with_prio.substr(0, pos);
      priority = worker_name_with_prio.substr(pos+1);
      RDMA_LOG(2) << "Connecting to remote node " << worker_name << " with priority " << priority;
    }
    else {
      worker_name_with_prio = p.first;
      worker_name = p.first;
      RDMA_LOG(2) << "Connecting to remote node " << worker_name;
    }
    RdmaChannel* rc = p.second;
    GetRemoteAddressRequest req;
    GetRemoteAddressResponse resp;
    // get the channel cache
    SharedGrpcChannelPtr client_channel =
        channel_cache_->FindWorkerChannel(worker_name_with_prio);
    GrpcVerbsClient* client = new GrpcVerbsClient(client_channel);
    CHECK(client != nullptr) << "No worker(+priority) known as " << worker_name_with_prio;

    // setting up request
    if (flow_priority)
      req.set_host_name(local_worker_ + "+" + priority);
    else
      req.set_host_name(local_worker_);
    Channel* channel_info = req.mutable_channel();
    channel_info->set_lid(rc->self_.lid);
    channel_info->set_qpn(rc->self_.qpn);
    channel_info->set_psn(rc->self_.psn);
    channel_info->set_snp(rc->self_.snp);
    channel_info->set_iid(rc->self_.iid);
    for (int i = 0; i < RdmaChannel::kNumMessageBuffers; i++) {
      MemoryRegion* mr = req.add_mr();
      mr->set_remote_addr(
          reinterpret_cast<uint64_t>(rc->message_buffers_[i]->buffer_));
      mr->set_rkey(rc->message_buffers_[i]->self_->rkey);
    }
    // synchronous call
    Status s;
    int attempts = 0;
    static const int max_num_attempts = 5;
    do {
      s = client->GetRemoteAddress(&req, &resp);
      // save obtained remote addresses
      // connect to the remote channel
      if (s.ok()) {
        if (flow_priority) {
          CHECK(resp.host_name().find(worker_name) != -1);
        }
        else {
          CHECK(worker_name.compare(resp.host_name()) == 0);
        }
        RdmaAddress ra;
        ra.lid = resp.channel().lid();
        ra.qpn = resp.channel().qpn();
        ra.psn = resp.channel().psn();
        ra.snp = resp.channel().snp();
        ra.iid = resp.channel().iid();
        rc->SetRemoteAddress(ra, false);
        if (flow_priority)
          rc->Connect(priority);
        else
          rc->Connect();
        int i = 0;
        int idx[] = {1, 0};
        for (const auto& mr : resp.mr()) {
          // the connections are crossed, i.e.
          // local tx_message_buffer <---> remote rx_message_buffer_
          // local rx_message_buffer <---> remote tx_message_buffer_
          // hence idx[] = {1, 0}.
          RdmaMessageBuffer* rb = rc->message_buffers_[idx[i]];
          RemoteMR rmr;
          rmr.remote_addr = mr.remote_addr();
          rmr.rkey = mr.rkey();
          rb->SetRemoteMR(rmr, false);
          i++;
        }
        CHECK(i == RdmaChannel::kNumMessageBuffers);
      } else {
        LOG(ERROR) << "Connecting to " << worker_name << ": Got "
                   << s.error_message() << ". Retrying (" << (attempts + 1)
                   << "/" << max_num_attempts << ")...";
        if (++attempts == max_num_attempts) {
          break;
        }
        worker_env_->env->SleepForMicroseconds(2000000);
      }
    } while (!s.ok());
    if (flow_priority)
      RDMA_LOG(0) << "Connected to remote node " << worker_name 
            << " with priority " << priority << " in queue pair number " << rc->self_.qpn;
    else
      RDMA_LOG(0) << "Connected to remote node " << worker_name
            << " in queue pair number " << rc->self_.qpn;
    delete client;
  }
}

// Check connectivity by pinging every channel
bool RdmaMgr::ConnectivityCheck() {
  int i, flow_cnt = 1, rcnt = 0, scnt = 0;

  for (const auto& p : channel_table_) {
    string worker_name = p.first;
    RdmaChannel* rc = p.second;

    VLOG(2) << "Ping to " << worker_name;
    CHECK(rc->PingPostSend() == 0) << "Couldn't post send  to " << worker_name
                                   << " with error: " << std::strerror(errno);
    for (i = 0; i < rc->adapter_->params_.queue_depth - 1; i++) {
      rc->Recv();
    }
  }

  if (flow_priority)
    flow_cnt = flow_priority;
  while (rcnt < flow_cnt * num_remote_workers_ || scnt < flow_cnt * num_remote_workers_) {
    int ne;
    do {
      ne = ibv_poll_cq(rdma_adapter_->cq_, 2 * flow_cnt * num_remote_workers_,
                       rdma_adapter_->wc_);
      CHECK(ne >= 0) << "poll CQ failed " << ne << "with error"
                     << std::strerror(errno);
    } while (ne < 1);

    for (i = 0; i < ne; ++i) {
      ibv_wc_status s = rdma_adapter_->wc_[i].status;
      // recv complete
      if ((int)rdma_adapter_->wc_[i].wr_id == RdmaChannel::kPingRecvWrid) {
        CHECK(s == IBV_WC_SUCCESS)
            << ": " << ibv_wc_status_str(rdma_adapter_->wc_[i].status) << "("
            << rdma_adapter_->wc_[i].status << ") for PING_RECV_WRID";
        ++rcnt;
        // send complete
      } else {
        RdmaChannel* rc =
            reinterpret_cast<RdmaChannel*>(rdma_adapter_->wc_[i].wr_id);
        CHECK(s == IBV_WC_SUCCESS)
            << ": " << ibv_wc_status_str(rdma_adapter_->wc_[i].status) << "("
            << rdma_adapter_->wc_[i].status << ") to " << rc->remote_name_;
        ++scnt;
      }
    }  // for
  }    // while
  CHECK(rcnt == scnt) << "Connectivity check failed!";
  rdma_adapter_->StartPolling();
  return (flow_cnt * num_remote_workers_ == rcnt) && (flow_cnt * num_remote_workers_ == scnt);
}

RdmaMgr::~RdmaMgr() {
  for (const auto& p : channel_table_) delete p.second;
  channel_table_.clear();
  delete rdma_adapter_;
}

// Find a channel via the given name.
// Args:
//   name: peer name, e.g. worker1
// Returns
//   channel object that is connected to the named peer.
RdmaChannel* RdmaMgr::FindChannel(const string& name) {
  ChannelTable::iterator iter = channel_table_.find(name);
  CHECK(iter != channel_table_.end());
  return iter->second;
}

bool IsGDRAvailable() {
#if defined(__APPLE__)
  return false;
#elif defined(PLATFORM_WINDOWS)
  return false;
#else
  std::ifstream ifs("/proc/modules");
  string line;
  while (std::getline(ifs, line)) {
    auto sep = line.find(' ');
    CHECK_NE(sep, std::string::npos);
    if (line.substr(0, sep) == "nv_peer_mem") {
      return true;
    }
  }
  return false;
#endif
}

int TryToReadNumaNode(ibv_device* device) {
#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node 0";
  return 0;
#elif defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#else
  VLOG(2) << "Trying to read NUMA node for device: " << device->name;
  static const int kUnknownNumaNode = -1;

  auto filename = string(device->ibdev_path) + "/device/numa_node";

  std::ifstream ifs(filename.c_str());
  string content;
  CHECK(std::getline(ifs, content));

  int32 value;
  if (strings::safe_strto32(content, &value)) {
    if (value < 0) {
      LOG(INFO) << "Successful NUMA node read from SysFS had negative value ("
                << value
                << "), but there must be at least one NUMA node"
                   ", so returning NUMA node zero";
      return 0;
    }
    LOG(INFO) << "NUMA node for device: " << device->name << " is " << value;
    return value;
  }
  return kUnknownNumaNode;
#endif
}

void MRDeleter(ibv_mr* mr) {
  if (mr) {
    ibv_dereg_mr(mr);
  }
}

void RdmaMgr::InitAllocators() {
  static std::once_flag flag;
  std::call_once(
      flag, [this]() { RdmaMemoryMgr::Singleton().pd_ = rdma_adapter_->pd_; });
}

/*static*/ void RdmaMgr::RegMemVisitors() {
  SubAllocator::Visitor alloc_visitor = [](void* ptr, int numa_node,
                                           size_t num_bytes) {
    RdmaMemoryMgr::Singleton().InsertMemoryRegion(
        ptr, num_bytes, strings::StrCat("CPU:", numa_node));
  };
  SubAllocator::Visitor free_visitor = [](void* ptr, int numa_node,
                                          size_t num_bytes) {
    RdmaMemoryMgr::Singleton().EvictMemoryRegion(ptr, num_bytes);
  };

  ProcessState::singleton()->AddCPUAllocVisitor(alloc_visitor);
  ProcessState::singleton()->AddCPUFreeVisitor(free_visitor);

#if GOOGLE_CUDA
  GPUProcessState::singleton()->AddGpuHostAllocVisitor(0, alloc_visitor);
  GPUProcessState::singleton()->AddGpuHostFreeVisitor(0, free_visitor);

  if (IsGDRAvailable()) {
    // Note we don't free allocated GPU memory so there is no free visitor

    // TODO: This is to fix the 'invalid use of member in static member function
    // bug'.
    //       Waiting for better implementation.
    //       int32_t bus_id = TryToReadNumaNode(rdma_adapter_->context_->device)
    //       + 1;
    int32_t bus_id = 0;

    SubAllocator::Visitor cuda_alloc_visitor = [](void* ptr, int gpu_id,
                                                  size_t num_bytes) {
      RdmaMemoryMgr::Singleton().InsertMemoryRegion(
          ptr, num_bytes, strings::StrCat("GPU:", gpu_id));
    };
    GPUProcessState::singleton()->AddGPUAllocVisitor(bus_id,
                                                     cuda_alloc_visitor);
    LOG(INFO) << "Instrumenting GPU allocator with bus_id " << bus_id;
  }
#endif  // GOOGLE_CUDA
}

}  // end namespace tensorflow

#endif
