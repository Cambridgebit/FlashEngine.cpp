#pragma once
#include "fe/model.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>

namespace fe {
struct OutputEvent {
  uint64_t request_id=0;std::vector<int64_t> token_ids;
  bool finished=false;std::string finish_reason;
};
struct GenerateRequest {std::vector<int64_t> token_ids;int64_t max_tokens=128;bool ignore_eos=false;};
class RequestHandle {
  std::shared_ptr<std::atomic<bool>> flag_;
public:
  uint64_t request_id=0;
  RequestHandle()=default;
  RequestHandle(uint64_t id,std::shared_ptr<std::atomic<bool>> flag):flag_(std::move(flag)),request_id(id){}
  void cancel()const{if(flag_)flag_->store(true);}
};
class Engine {
  struct State;
  Config config_;std::unique_ptr<DeviceModel> model_;Scheduler scheduler_;
  std::mutex mutex_;std::condition_variable cv_;std::deque<std::shared_ptr<State>> pending_;
  std::map<uint64_t,std::shared_ptr<State>> states_;std::atomic<uint64_t> next_{1},inflight_{0};
  bool stopping_=false;std::thread driver_;Json snapshot_,metrics_,recent_=Json::array();
  std::once_flag stop_once_;
  void loop();void emit(const std::shared_ptr<State>& state);
  void refresh();
public:
  using Sink=std::function<void(OutputEvent)>;
  Engine(Config config,std::unique_ptr<DeviceModel> model);
  ~Engine();
  void stop();
  RequestHandle submit(GenerateRequest request,Sink sink);
  Json runtime_snapshot();
  const Config& config()const{return config_;}
};
}
