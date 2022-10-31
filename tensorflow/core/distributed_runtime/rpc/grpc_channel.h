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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "grpcpp/grpcpp.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include <time.h>
#include "tensorflow/core/util/device_name_utils.h"
#include <sys/socket.h>    
#include <netinet/in.h>  
#include <arpa/inet.h> 
#include "rapidjson/prettywriter.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/memorystream.h"
#include "re2/re2.h"

namespace tensorflow {

// Consolidated parameter structure to ease use of generic interfaces.
//
// Each job_id requires:
// - a list of host:port (or sparse list of index:host:port)
// - the number of tasks per replica
class GrpcChannelSpec {
 public:
  struct HostPortsJob {
    HostPortsJob(const string& job_id, const std::map<int, string>& host_ports)
        : job_id(job_id), host_ports(host_ports) {}
    const string job_id;
    const std::map<int, string> host_ports;
  };

  Status AddHostPortsJob(const string& job_id,
                         const std::vector<string>& host_ports);

  Status AddHostPortsJob(const string& job_id,
                         const std::map<int, string>& host_ports);

  const std::vector<HostPortsJob>& host_ports_jobs() const {
    return host_ports_jobs_;
  }

 private:
  std::vector<HostPortsJob> host_ports_jobs_;
  std::set<string> job_ids_;
};

class GrpcChannelCache {
 public:
  virtual ~GrpcChannelCache() {}

  // Populates *workers with names of all workers which this object
  // was created to handle.  Worker names are in the format
  //  /job:<job identifier>/task:<task id>
  // e.g. /job:mnist/task:2
  virtual void ListWorkers(std::vector<string>* workers) = 0;
  virtual void ListWorkersInJob(const string& job_name,
                                std::vector<string>* workers) = 0;

  // If found, returns a gRPC channel that is connected to the remote
  // worker named by 'target'. 'target' is of the following
  // format: /job:<job identifier>/task:<task id>
  // E.g., /job:mnist/task:2
  virtual SharedGrpcChannelPtr FindWorkerChannel(const string& target) = 0;

  // Translates a string in the form `/job:X/task:Z` into a host_port.
  virtual string TranslateTask(const string& task) = 0;
};

typedef std::function<SharedGrpcChannelPtr(string)> ChannelCreationFunction;

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& channel_spec,
                                      ChannelCreationFunction channel_func);

// Below here are internal-only functions.

::grpc::ChannelArguments GetChannelArguments(const RPCOptions* rpc_options);

ChannelCreationFunction ConvertToChannelCreationFunction(
    const std::function<Status(string, const RPCOptions*,
                               SharedGrpcChannelPtr*)>& new_channel_func_ptr);

Status NewHostPortGrpcChannel(const string& target,
                              const RPCOptions* rpc_options,
                              SharedGrpcChannelPtr* channel_pointer);

// Above here are internal-only functions.

// for Geryon
extern mutex allocating_mu;
extern bool finish_allocating_;
extern bool is_worker_;
extern bool is_chief_;
extern int64 last_step_id_;
extern int64 steps_;
extern struct timespec current_tv, last_tv;
extern int64 iteration_time_;
extern std::unordered_map<int, int64> ps_step_id_pool_;
extern std::unordered_map<int, int64> worker_step_id_pool_;
extern std::map<std::pair<string, int>, string> ip_table_;
extern std::map<string, int> socket_table_;
extern uint32_t mapping_mode;
extern uint8_t flow_priority;
extern std::vector<uint8_t> priority_vec;

typedef std::vector<string> TensorOrder;
extern TensorOrder model_parameter_order_;
extern TensorOrder parameter_order_;
extern TensorOrder gradient_order_;

typedef std::unordered_map<string, size_t> PriorityTable;
extern PriorityTable parameter_table_;
extern PriorityTable gradient_table_;
extern std::unordered_map<size_t, PriorityTable> parameter_table_map_;
extern std::unordered_map<size_t, PriorityTable> gradient_table_map_;
extern pthread_rwlock_t gradient_table_rwlock_;
struct MetaPriorityInfo {
  bool finished;  // finish_allocating
  size_t size;  // the number of table entries
  bool is_last_table;  // the last table?
};

extern struct timespec ccp_start_tv_;
extern bool should_profile_;
extern std::unordered_map<string, bool> edges_set_;
extern std::unordered_map<string, std::vector<size_t>> worker_power_vector_;
extern std::unordered_map<string, size_t> worker_power_;
extern std::unordered_map<size_t, std::vector<size_t>> powergrade_to_worker_;
extern std::unordered_map<size_t, size_t> worker_to_powergrade_;
extern string last_egde_name;

// bayesian optmization module
struct MetaPMInfo {
  int num_heter_workers;  // the number of heterogeneous workers
  int num_priorities;  // the number of priorities
  int num_parameters;  // the number of parameter layers
};
extern int para_map_sockfd_;
extern int num_heter_workers_ ;
extern std::unordered_map<string, double> probe_point_;


bool is_new_iteration(bool is_worker, int64 step_id, 
                      int64 last_step_id, string src_name, string edge_name) {
  if (is_worker) {  
    if (step_id != last_step_id) {
      string var_name;
      RE2::PartialMatch(edge_name, "edge_\\d+_(.+)", &var_name);
      std::vector<string>::iterator it = find(parameter_order_.begin(), parameter_order_.end(), var_name);
 
      if (it != parameter_order_.end()) {
        last_step_id = step_id;
        return true;
      }
    }
    return false;
  }
  else {
    if (str_util::StrContains(edge_name, "gradients")) {
      DeviceNameUtils::ParsedName pn;
      decltype(ps_step_id_pool_)::iterator iter;
      bool success;
      DeviceNameUtils::ParseFullName(src_name, &pn);
      std::tie(iter, success) = ps_step_id_pool_.insert(
          std::make_pair(pn.task, step_id));
      if (!success && step_id != ps_step_id_pool_[pn.task]) {
        ps_step_id_pool_.clear();
        ps_step_id_pool_.insert(std::make_pair(pn.task, step_id));
        return true;
      }
    }
    return false;
  }
}

void steps_sync(bool is_worker, bool is_chief) {
  if (is_worker && is_chief) {
    char send_buf[2];
    strcpy(send_buf, std::to_string(steps_).c_str());
    for (auto s : socket_table_) {
      int ret = send(s.second, send_buf, 2, 0);
    }
  }
  else {
    char recv_buf[2];
    struct timeval timeout={2,0}; //2s
    int ret=setsockopt(socket_table_["worker0"],SOL_SOCKET,SO_RCVTIMEO,(const char*)&timeout,sizeof(timeout));
    int recv_bytes = 0;
    int last_recv_bytes;
    while (recv_bytes >= 0) {
      last_recv_bytes = recv_bytes;
      recv_bytes = recv(socket_table_["worker0"], recv_buf, 2, 0);
    }
    if (last_recv_bytes > 0) {
      steps_ = stoi(string(recv_buf));
    }
  }
}

// Function to get environment variable
// Args:
//    var_name - the name of the environmental variable
// Returns:
//    string with it's value or empty string if not set
string get_env_var(char const* var_name) {
  char const* var_temp = getenv(var_name);

  return (var_temp == NULL) ? string() : string(var_temp);
}

// set the default or environment value to the configuration parameter.
// Args:
//   default_val- the default value for this parameter
//   env_param- the environment parameter's name
// Returns:
//   32-bit value
uint32_t set_param(uint32_t default_val, const char* env_param) {
  uint32_t val = default_val;
  string val_s;

  val_s = get_env_var(env_param);

  if (!val_s.empty()) {
    val = stoi(val_s);
  }
  return val;
}

std::vector<uint8_t> split_and_convert_uint8(const string &str, char delim) {
  std::vector<uint8_t> elems;
  std::istringstream iss(str);
  for (string item; getline(iss, item, delim); )
    if (item.empty())
      continue;
    else
      elems.push_back(atoi(item.c_str()));
  return elems;
}

std::vector<double> split_and_convert_double(const string &str, char delim) {
  std::vector<double> elems;
  std::istringstream iss(str);
  for (string item; getline(iss, item, delim); )
    if (item.empty())
      continue;
    else
      elems.push_back(atof(item.c_str()));
  return elems;
}

string unordered_map2string(const std::unordered_map<string, size_t> &m) {
  rapidjson::Document document; 
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator(); 
  rapidjson::Value root(rapidjson::kObjectType); 
  rapidjson::Value key(rapidjson::kStringType); 
  for( std::unordered_map<string, size_t>::const_iterator it = m.begin(); it != m.end(); ++it) {
    key.SetString(it->first.c_str(), allocator); 
    root.AddMember(key, it->second, allocator);
  }
  rapidjson::StringBuffer buffer; 
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); 
  root.Accept(writer); 
  return buffer.GetString(); 
} 

std::unordered_map<string, size_t> string2unordered_map(const string& jsonString) {
  rapidjson::Document document;
  std::unordered_map<string, size_t> m;
  document.SetObject();
  document.Parse<rapidjson::kParseDefaultFlags>(jsonString.c_str());

  for(rapidjson::Value::ConstMemberIterator iter = document.MemberBegin(); iter != document.MemberEnd(); ++iter) {
    const char * key = iter->name.GetString();
    const rapidjson::Value& val = iter->value;
    m.insert(std::make_pair(key, val.GetInt64()));  
  }

  return m;
}

std::unordered_map<string, double> string2unordered_map_double(const string& jsonString) {
  rapidjson::Document document;
  std::unordered_map<string, double> m;
  document.SetObject();
  document.Parse<rapidjson::kParseDefaultFlags>(jsonString.c_str());

  for(rapidjson::Value::ConstMemberIterator iter = document.MemberBegin(); iter != document.MemberEnd(); ++iter) {
    const char * key = iter->name.GetString();
    const rapidjson::Value& val = iter->value;
    m.insert(std::make_pair(key, val.GetDouble()));  
  }

  return m;
}

size_t FindPriority(const string& name, bool is_worker, string src_name) {
  string simple_edge_name;
  size_t priority;
  RE2::PartialMatch(name, "edge_\\d+_(.+)", &simple_edge_name);
  PriorityTable::iterator iter;
  if (is_worker) {
    iter = parameter_table_.find(simple_edge_name);
    if (iter == parameter_table_.end())
      priority = priority_vec.back();
    else
      priority = iter->second;
    return priority;
  }
  else {
    DeviceNameUtils::ParsedName pn;
    DeviceNameUtils::ParseFullName(src_name, &pn);
    decltype(worker_to_powergrade_)::iterator power_iter;
    pthread_rwlock_rdlock(&gradient_table_rwlock_);
    power_iter = worker_to_powergrade_.find(pn.task);
    if (power_iter != worker_to_powergrade_.end()) {
      size_t powergrade = power_iter->second;
      decltype(gradient_table_map_)::iterator map_iter;
      map_iter = gradient_table_map_.find(powergrade);

      if (map_iter != gradient_table_map_.end()) {
        iter = map_iter->second.find(simple_edge_name);
        if (iter == map_iter->second.end()) {
          priority = priority_vec.back();
        }
        else {
          priority = iter->second;
        } 
      }
    }
    else {
      priority = priority_vec.back();
    }
    pthread_rwlock_unlock(&gradient_table_rwlock_);
    return priority;
  }
}

size_t sum_vector(std::vector<size_t> v) {
  return std::accumulate(std::begin(v), std::end(v), 0);
}

void send_powergrade(std::unordered_map<size_t, size_t> worker_to_powergrade) {
  char meta_powergrade_buf[8];
  std::unordered_map<string, size_t> tmp_powergrade;
  for (auto w: worker_to_powergrade)
    tmp_powergrade.insert(std::make_pair(std::to_string(w.first), w.second));

  string powergrade_json = unordered_map2string(tmp_powergrade);
  size_t str_size = powergrade_json.size();
  memcpy(meta_powergrade_buf, &str_size, sizeof(str_size));
  char *powergrade_buf = (char*)malloc(str_size + 1);
  CHECK(powergrade_buf) << "Failed to malloc powergrade buffer.";
  strcpy(powergrade_buf, powergrade_json.c_str());

  int ret;
  for (auto s : socket_table_) {
    ret = send(s.second, meta_powergrade_buf, 8, 0);
    CHECK(ret != -1) << "Failed to send meta powergrade information.";
    ret = send(s.second, powergrade_buf, str_size + 1, 0);
    CHECK(ret != -1) << "Failed to send powergrade information.";
    //VLOG(INFO) << "send powergrade to " << s.first;
  }
}

void recv_powergrade(std::unordered_map<size_t, size_t>& worker_to_powergrade ) {
  size_t str_size;
  char meta_recv_buf[8];
  int meta_recv_bytes = 0;
  while (meta_recv_bytes <= 0) {
    meta_recv_bytes = recv(socket_table_["worker0"], meta_recv_buf, 8, 0);
  }
  memcpy(&str_size, meta_recv_buf, sizeof(str_size));

  char *recv_buf = (char*)malloc(str_size + 1);
  CHECK(recv_buf) << "Failed to malloc receiver's buffer.";

  int recv_bytes = 0;
  int total_recv_bytes = 0;
  while (total_recv_bytes < str_size + 1) {
    recv_bytes = recv(socket_table_["worker0"], recv_buf+total_recv_bytes, str_size + 1 - total_recv_bytes, 0);
    total_recv_bytes += recv_bytes;
  }

  std::unordered_map<string, size_t> tmp_powergrade;
  tmp_powergrade = string2unordered_map(string(recv_buf));
  for (auto w: tmp_powergrade) {
    worker_to_powergrade.insert(std::make_pair(stoi(w.first), w.second));
  }
}

void worker_power_sync(bool is_worker, bool is_chief) {
  if (is_chief) {
    char buffer[1024];
    if (!is_worker) {
      for (auto w : worker_power_vector_)
        worker_power_.insert(std::make_pair(w.first, sum_vector(w.second)));
      string worker_power_json = unordered_map2string(worker_power_);
      strcpy(buffer, worker_power_json.c_str());
      int ret = send(socket_table_["worker0"], buffer, worker_power_json.size() + 1, 0);
      CHECK(ret != -1) << "Failed to send worker power information.";
    }
    else {
      int recv_bytes = 0;
      while (recv_bytes <= 0) {
        if (recv_bytes == -1)
          perror("recv_bytes == -1");
        recv_bytes = recv(socket_table_["ps0"], buffer, 1024, 0);
      }
      worker_power_ = string2unordered_map(string(buffer));
    }
  }
}

int cluster_worker_power(bool is_worker, bool is_chief, 
                        std::unordered_map<string, size_t> worker_power, double gap) {
  if (is_worker && is_chief) {
    string str_powergrade = get_env_var("SPECIFY_POWER");
    if (str_powergrade.empty() == false) {
      std::vector<uint8_t> powergrade_vec = split_and_convert_uint8(str_powergrade, ',');
      for (int i = 0; i < powergrade_vec.size(); i++) {
        worker_to_powergrade_[(size_t)i] = (size_t)powergrade_vec[i];
        powergrade_to_worker_[(size_t)powergrade_vec[i]].push_back((size_t)i);
      }
      send_powergrade(worker_to_powergrade_);

      std::vector<size_t> grade_vec;
      for (auto w : worker_to_powergrade_) {
        std::vector<size_t>::iterator it = find(grade_vec.begin(), grade_vec.end(), w.second);
        if (it == grade_vec.end())
          grade_vec.push_back(w.second);
      }

      return grade_vec.size();
    }
    else if (set_param(0, "DIST_STRAG") != 0) {
      std::vector<std::pair<string, size_t>> tmp;
      for (auto& i : worker_power)
        tmp.push_back(i);

      std::sort(tmp.begin(), tmp.end(), 
                [=](std::pair<string, size_t>& a, std::pair<string, size_t>& b) { 
                      return a.second < b.second; 
                });

      size_t tmp_power = tmp[0].second;
      int grade = 0;
      decltype(powergrade_to_worker_)::iterator iter;
      bool success;
      for (auto& i : tmp) {
        if (i.second > tmp_power*(1+gap)) {
          tmp_power = i.second;
          grade++;
        }
        std::tie(iter, success) = powergrade_to_worker_.insert(
            std::pair<size_t, std::vector<size_t>>(grade, {(size_t)stoi(i.first)} ));
        if (!success)
          powergrade_to_worker_[grade].push_back((size_t)stoi(i.first));
        worker_to_powergrade_[(size_t)stoi(i.first)] = grade;
      }

      send_powergrade(worker_to_powergrade_);
      return grade + 1;
    }
    else {
      for (auto& i : worker_power) {
        powergrade_to_worker_[0].push_back((size_t)stoi(i.first));
        worker_to_powergrade_[(size_t)stoi(i.first)] = 0;
      }

      send_powergrade(worker_to_powergrade_);
      return 1;
    }
  }
  else {
    recv_powergrade(worker_to_powergrade_);
    std::vector<size_t> grade_vec;
    for (auto w : worker_to_powergrade_) {
      std::vector<size_t>::iterator it = find(grade_vec.begin(), grade_vec.end(), w.second);
      if (it == grade_vec.end())
        grade_vec.push_back(w.second);
    }

    return grade_vec.size();
  }
}

void send_allocating_results(bool finish_allocating, int64 iteration_time) {
  MetaPriorityInfo parameter_info, gradient_info;
  char meta_parameter_buf[24], meta_gradient_buf[24];

  int ret;
  for (auto s : socket_table_) {
    if (str_util::StrContains(s.first, "worker")) {
      size_t worker_id = stoi(s.first.substr(6));
      size_t powergrade = iteration_time == -2 ? 0 : worker_to_powergrade_[worker_id];
      string prameter_json = unordered_map2string(parameter_table_map_[powergrade]);
      parameter_info.finished = finish_allocating;
      parameter_info.size = prameter_json.size();
      parameter_info.is_last_table = true;
      memcpy(meta_parameter_buf, &parameter_info, sizeof(parameter_info));
      char *parameter_buf = (char*)malloc(parameter_info.size + 1);
      CHECK(parameter_buf) << "Failed to malloc parameter buffer.";
      strcpy(parameter_buf, prameter_json.c_str());

      ret = send(s.second, meta_parameter_buf, 24, 0);
      CHECK(ret != -1) << "Failed to send meta parameter information.";
      ret = send(s.second, parameter_buf, parameter_info.size + 1, 0);
      CHECK(ret != -1) << "Failed to send parameter information.";
    }
    else {
      for (size_t i = 0; i < num_heter_workers_; i++) {
        pthread_rwlock_rdlock(&gradient_table_rwlock_);
        string gradient_json = unordered_map2string(gradient_table_map_[i]);
        pthread_rwlock_unlock(&gradient_table_rwlock_);
        gradient_info.finished = finish_allocating;
        gradient_info.size = gradient_json.size();
        gradient_info.is_last_table = i == (num_heter_workers_ - 1) ? true : false;
        memcpy(meta_gradient_buf, &gradient_info, sizeof(gradient_info));
        char *gradient_buf = (char*)malloc(gradient_info.size + 1);
        CHECK(gradient_buf) << "Failed to malloc gradient buffer.";
        strcpy(gradient_buf, gradient_json.c_str());

        ret = send(s.second, meta_gradient_buf, 24, 0);
        CHECK(ret != -1) << "Failed to send meta gradient information.";
        ret = send(s.second, gradient_buf, gradient_info.size + 1, 0);
        CHECK(ret != -1) << "Failed to send gradient information.";
      }
    }
  }

  size_t powergrade = worker_to_powergrade_[0];
  parameter_table_ = parameter_table_map_[powergrade];
}

bool recv_allocating_result(PriorityTable& tmp_table, bool& is_last_table) {
  MetaPriorityInfo recv_info;
  char meta_recv_buf[24];
  int meta_recv_bytes = 0;
  while (meta_recv_bytes <= 0) {
    meta_recv_bytes = recv(socket_table_["worker0"], meta_recv_buf, 24, 0);
  }
  memcpy(&recv_info, meta_recv_buf, sizeof(recv_info));

  is_last_table = recv_info.is_last_table;
  char *recv_buf = (char*)malloc(recv_info.size + 1);
  CHECK(recv_buf) << "Failed to malloc receiver's buffer.";

  int recv_bytes = 0;
  int total_recv_bytes = 0;
  while (total_recv_bytes < recv_info.size + 1) {
    recv_bytes = recv(socket_table_["worker0"], recv_buf+total_recv_bytes, recv_info.size + 1 - total_recv_bytes, 0);
    total_recv_bytes += recv_bytes;
  }

  tmp_table = string2unordered_map(string(recv_buf));
  return recv_info.finished;
}

bool layer_function_allocating(int64 steps, int64 iteration_time) {
  size_t prio_index;

  std::unordered_map<string, size_t> tmp_parameter_table, tmp_gradient_table;
  tmp_parameter_table.clear();
  tmp_gradient_table.clear();
  for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
    string var = model_parameter_order_[model_para_index];

    if (str_util::StrContains(var, "conv"))
      prio_index = priority_vec.size() - 2;
    else
      prio_index = priority_vec.size() - 1;

    for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
      if (str_util::StrContains(parameter_order_[para_index], var)) {
        tmp_parameter_table.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
        tmp_gradient_table.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
      }
    }
  }
  pthread_rwlock_wrlock(&gradient_table_rwlock_);
  for (size_t i = 0; i < num_heter_workers_; i++) {
    parameter_table_map_.insert(std::make_pair(i, tmp_parameter_table));
    gradient_table_map_.insert(std::make_pair(i, tmp_gradient_table));
  }
  pthread_rwlock_unlock(&gradient_table_rwlock_);

  if (steps >= 13)
    return true;
  else
    return false;

}

bool equal_allocating(int64 steps, int64 iteration_time) {
  size_t num_parameters = model_parameter_order_.size();
  size_t parameter_group_size = num_parameters / flow_priority;

  std::unordered_map<string, size_t> tmp_parameter_table, tmp_gradient_table;
  tmp_parameter_table.clear();
  tmp_gradient_table.clear();
  for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
    size_t prio_index = model_para_index / parameter_group_size;
    prio_index = prio_index >= flow_priority ? flow_priority - 1 : prio_index;

    string var = model_parameter_order_[model_para_index];
    for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
      if (str_util::StrContains(parameter_order_[para_index], var)) {
        tmp_parameter_table.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
        tmp_gradient_table.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
      }
    }
  }
  pthread_rwlock_wrlock(&gradient_table_rwlock_);
  for (size_t i = 0; i < num_heter_workers_; i++) {
    parameter_table_map_.insert(std::make_pair(i, tmp_parameter_table));
    gradient_table_map_.insert(std::make_pair(i, tmp_gradient_table));
  }
  pthread_rwlock_unlock(&gradient_table_rwlock_);

  if (steps >= 13)
    return true;
  else
    return false;
}

bool specify_allocating(int64 steps, int64 iteration_time) {
  std::vector<double> priority_threshold_vec_;

  if (iteration_time == -2) {
    size_t num_parameters = model_parameter_order_.size();
    size_t parameter_group_size = num_parameters / flow_priority;

    for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
      size_t prio_index = model_para_index / parameter_group_size;
      prio_index = prio_index >= flow_priority ? flow_priority - 1 : prio_index;

      string var = model_parameter_order_[model_para_index];
      for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
        if (str_util::StrContains(parameter_order_[para_index], var)) {
          parameter_table_.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
          gradient_table_.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
        }
      }
    }

    pthread_rwlock_wrlock(&gradient_table_rwlock_);
    for (size_t i = 0; i < num_heter_workers_; i++) {
      parameter_table_map_.insert(std::make_pair(i, parameter_table_));
      gradient_table_map_.insert(std::make_pair(i, gradient_table_));
    }
    pthread_rwlock_unlock(&gradient_table_rwlock_);
  }
  else {
    priority_threshold_vec_ = split_and_convert_double(get_env_var("PRIORITY_THRESHOLD"), ',');
    CHECK(priority_threshold_vec_.size()>0) << "You did not define PRIORITY_THRESHOLD.";
    std::cout << "threshold vector: ";
      for (auto p : priority_threshold_vec_)
        std::cout << p << "\t";
      std::cout << std::endl;
    size_t num_parameters = model_parameter_order_.size();
    size_t prio_index;
    std::unordered_map<string, size_t> tmp_parameter_table, tmp_gradient_table;
    for (int i = 0; i < num_heter_workers_; i++) {
      tmp_parameter_table.clear();
      tmp_gradient_table.clear();
      for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
        string var = model_parameter_order_[model_para_index];
        double pos = (double) model_para_index / model_parameter_order_.size();
        bool found = false;
        for (int j = i*(priority_vec.size()-1); j < (i+1)*(priority_vec.size()-1); j++)
          if (pos < priority_threshold_vec_[j]) {
            prio_index = j - i*(priority_vec.size()-1);
            found = true;
            break;
          }
        if (!found)
          prio_index = priority_vec.size() - 1;
      
        //VLOG(INFO) << i << "\t" << pos << "\t" << int(priority_vec[prio_index]) << "\t" << var;
        for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
          if (str_util::StrContains(parameter_order_[para_index], var)) {
            tmp_parameter_table.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
            tmp_gradient_table.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
          }
        }
      }
      pthread_rwlock_wrlock(&gradient_table_rwlock_);
      parameter_table_map_.insert(std::make_pair((size_t)i, tmp_parameter_table));
      gradient_table_map_.insert(std::make_pair((size_t)i, tmp_gradient_table));
      pthread_rwlock_unlock(&gradient_table_rwlock_);
    }
  }
  
  if (steps >= 13)
    return true;
  else
    return false;
}

bool bo_allocating(int64 steps, int64 iteration_time) {
  bool finish_allocating = false;
  std::vector<double> priority_threshold_vec_;

  if (iteration_time < 0) {
    MetaPMInfo parameter_mapping_info;
    size_t num_parameters = model_parameter_order_.size();
    size_t parameter_group_size = num_parameters / flow_priority;

    for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
      size_t prio_index = model_para_index / parameter_group_size;
      prio_index = prio_index >= flow_priority ? flow_priority - 1 : prio_index;

      string var = model_parameter_order_[model_para_index];
      for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
        if (str_util::StrContains(parameter_order_[para_index], var)) {
          parameter_table_.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
          gradient_table_.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
        }
      }
    }

    pthread_rwlock_wrlock(&gradient_table_rwlock_);
    for (size_t i = 0; i < num_heter_workers_; i++) {
      parameter_table_map_.insert(std::make_pair(i, parameter_table_));
      gradient_table_map_.insert(std::make_pair(i, gradient_table_));
    }
    pthread_rwlock_unlock(&gradient_table_rwlock_);
      
    if (iteration_time == -1) {
      parameter_mapping_info.num_heter_workers = num_heter_workers_;
      parameter_mapping_info.num_priorities = priority_vec.size();
      parameter_mapping_info.num_parameters = model_parameter_order_.size();
      char meta_parameter_mapping_buf[12];
      memcpy(meta_parameter_mapping_buf, &parameter_mapping_info, sizeof(parameter_mapping_info));
      int ret = send(para_map_sockfd_, meta_parameter_mapping_buf, 12, 0);
      CHECK(ret != -1) << "Failed to send meta parameter mapping information.";
    }

    return finish_allocating;
  }
  else {
    char send_buf[8];
    memcpy(send_buf, &iteration_time, sizeof(iteration_time));
    int ret = send(para_map_sockfd_, send_buf, 8, 0);
    CHECK(ret != -1) << "Failed to send iteration information.";

    char recv_buf[1024];
    memset(recv_buf, 0, 1024);
    ret = recv(para_map_sockfd_, recv_buf, 1024, 0);
    probe_point_ = string2unordered_map_double(string(recv_buf));
    if (probe_point_["finish"] > 0.5)
      finish_allocating = true;

    priority_threshold_vec_.clear();
    for (int i = 0; i < num_heter_workers_; i++)
      for (int j = 0; j < priority_vec.size() - 1; j++) {
        if (j == 0)
          priority_threshold_vec_.push_back(
              probe_point_["p"+std::to_string(i*(priority_vec.size()-1))]);
        else
          priority_threshold_vec_.push_back(
              priority_threshold_vec_[i*(priority_vec.size()-1)+j-1]
              + probe_point_["p"+std::to_string(i*(priority_vec.size()-1)+j)]
              * (1 - priority_threshold_vec_[i*(priority_vec.size()-1)+j-1]));
      }
    
    std::cout << "threshold vector: ";
    for (auto p : priority_threshold_vec_)
      std::cout << p << "\t";
    std::cout << std::endl;

    size_t num_parameters = model_parameter_order_.size();
    size_t prio_index;
    std::unordered_map<string, size_t> tmp_parameter_table, tmp_gradient_table;
    for (int i = 0; i < num_heter_workers_; i++) {
      tmp_parameter_table.clear();
      tmp_gradient_table.clear();
      for(size_t model_para_index = 0; model_para_index < model_parameter_order_.size(); model_para_index++) {
        string var = model_parameter_order_[model_para_index];
        double pos = (double) model_para_index / model_parameter_order_.size();
        bool found = false;
        for (int j = i*(priority_vec.size()-1); j < (i+1)*(priority_vec.size()-1); j++)
          if (pos < priority_threshold_vec_[j]) {
            prio_index = j - i*(priority_vec.size()-1);
            found = true;
            break;
          }
        if (!found)
          prio_index = priority_vec.size() - 1;
      
        //VLOG(INFO) << i << "\t" << pos << "\t" << int(priority_vec[prio_index]) << "\t" << var;
        for(size_t para_index = 0; para_index < parameter_order_.size(); para_index++) {
          if (str_util::StrContains(parameter_order_[para_index], var)) {
            tmp_parameter_table.insert(std::make_pair(parameter_order_[para_index], priority_vec[prio_index]));
            tmp_gradient_table.insert(std::make_pair(gradient_order_[para_index], priority_vec[prio_index]));
          }
        }
      }
      pthread_rwlock_wrlock(&gradient_table_rwlock_);
      parameter_table_map_.insert(std::make_pair((size_t)i, tmp_parameter_table));
      gradient_table_map_.insert(std::make_pair((size_t)i, tmp_gradient_table));
      pthread_rwlock_unlock(&gradient_table_rwlock_);
    }

    return finish_allocating;
  } 
}

bool allocating_and_distribution(int64 steps, int64 iteration_time, bool is_worker, bool is_chief) {
  bool finish_allocating = false;

  pthread_rwlock_wrlock(&gradient_table_rwlock_);
  parameter_table_map_.clear();
  gradient_table_map_.clear();
  pthread_rwlock_unlock(&gradient_table_rwlock_);
  parameter_table_.clear();
  gradient_table_.clear();

  if (is_worker && is_chief) {
    switch (mapping_mode) {
      case 0:
        finish_allocating = layer_function_allocating(steps, iteration_time);
        break;
      case 1:
        finish_allocating = equal_allocating(steps, iteration_time);
        break;
      case 2:
        finish_allocating = specify_allocating(steps, iteration_time);
        break;
      case 3:
        finish_allocating = bo_allocating(steps, iteration_time);
        break;
      default:
        CHECK(0) << "MAPPING_MODE only allows 0 ~ 3 by now.";
        break;
    }
    send_allocating_results(finish_allocating, iteration_time);
    /* if (finish_allocating)
      for (auto &p : parameter_table_map_[0])
        VLOG(INFO) << p.second << "\t" << p.first; */
  }
  else {
    if (is_worker) {
      bool is_last_table = true;
      finish_allocating = recv_allocating_result(parameter_table_, is_last_table);
    }
    else {
      bool is_last_table = false;
      size_t i = 0;
      while (!is_last_table) {
        finish_allocating = recv_allocating_result(gradient_table_, is_last_table);
        pthread_rwlock_wrlock(&gradient_table_rwlock_);
        gradient_table_map_.insert(std::make_pair(i, gradient_table_));
        pthread_rwlock_unlock(&gradient_table_rwlock_);
        i++;
      }
    }
  }

  return finish_allocating; 
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_H_
