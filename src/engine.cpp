#include "fe/engine.hpp"
#include <chrono>
#include <iostream>
#include <optional>

namespace fe {
using Clock=std::chrono::steady_clock;
struct Engine::State {
  Seq seq;Sink sink;size_t emitted=0;Clock::time_point submitted=Clock::now();std::optional<Clock::time_point> started;double ttft=-1;
};
Engine::Engine(Config config,std::unique_ptr<DeviceModel> model):config_(std::move(config)),model_(std::move(model)),scheduler_(config_){
  bind_model(scheduler_,*model_);
  for(auto name:{"generated_tokens_total","engine_steps_total","prefill_steps_total","decode_steps_total","scheduler_preemptions_total","spec_rounds_total","spec_committed_tokens_total","prefix_requests_total","prefix_hit_requests_total","prefix_cached_blocks_total","prefix_new_blocks_total","cancelled_requests_total","requests_total","ttft_seconds_sum","ttft_seconds_count"})metrics_[std::string("flashengine_")+name]=0;
  metrics_["flashengine_spec_draft_tokens_total"]=0;metrics_["flashengine_spec_accepted_tokens_total"]=0;
  for(int p=1;p<=config_.spec_num_tokens;++p){auto key="flashengine_spec_position_"+std::to_string(p);metrics_[key+"_proposed_total"]=0;metrics_[key+"_accepted_total"]=0;}
  metrics_["flashengine_engine_busy_seconds_total"]=0.;metrics_["flashengine_graph_fallbacks_total"]=0;refresh();std::cerr<<"[INFO ] resolved_config="<<config_.json().dump()<<'\n';driver_=std::thread([this]{loop();});
}
Engine::~Engine(){stop();}
void Engine::stop(){std::call_once(stop_once_,[this]{
  {std::lock_guard lock(mutex_);stopping_=true;for(auto& state:pending_)state->seq->cancelled->store(true);}
  cv_.notify_all();if(driver_.joinable())driver_.join();
});}
RequestHandle Engine::submit(GenerateRequest request,Sink sink){
  if(request.token_ids.empty()||request.max_tokens<1||request.token_ids.size()>=size_t(config_.max_model_len))throw InputError("非法 prompt/输出长度");
  if((request.token_ids.size()+config_.kvcache_block_size-1)/config_.kvcache_block_size>size_t(config_.num_kvcache_blocks))throw InputError("prompt 超过 KV 容量");
  std::lock_guard lock(mutex_);if(stopping_)throw InputError("服务正在停止");if(inflight_.load()>=uint64_t(config_.max_num_seqs*9))throw std::length_error("请求队列已满");
  uint64_t id=next_.fetch_add(1);auto state=std::make_shared<State>();state->seq=std::make_shared<Sequence>(id,std::move(request.token_ids),request.max_tokens);state->seq->ignore_eos=request.ignore_eos;state->sink=std::move(sink);pending_.push_back(state);++inflight_;cv_.notify_one();return {id,state->seq->cancelled};
}
Json Engine::runtime_snapshot(){std::lock_guard lock(mutex_);return snapshot_;}
void Engine::refresh(){
  auto sched=scheduler_.snapshot(),runtime=model_->snapshot();
  metrics_["flashengine_running_requests"]=sched["running"];metrics_["flashengine_waiting_requests"]=sched["waiting"];metrics_["flashengine_active_requests"]=sched["active"];
  metrics_["flashengine_kv_blocks_used"]=sched["used_blocks"];metrics_["flashengine_kv_blocks_total"]=sched["total_blocks"];metrics_["flashengine_scheduler_preemptions_total"]=sched["preemptions"];
  metrics_["flashengine_kv_used_blocks"]=sched["used_blocks"];metrics_["flashengine_kv_total_blocks"]=sched["total_blocks"];
  for(std::string name:{"prefix_requests","prefix_hit_requests","prefix_cached_blocks","prefix_new_blocks"})metrics_["flashengine_"+name+"_total"]=sched[name];
  metrics_["flashengine_prefix_resident_blocks"]=sched["prefix_resident_blocks"];metrics_["flashengine_prefix_evictions_total"]=sched["prefix_evictions"];
  metrics_["flashengine_partial_prefill_requests"]=sched["partial_prefills"];
  for(std::string name:{"steps","prefill_steps","decode_steps","prefill_tokens","decode_tokens","chunked_prefill_steps","admitted_requests"})metrics_["flashengine_scheduler_"+name+"_total"]=sched[name];
  if(runtime.contains("prefix_state")){auto& state=runtime["prefix_state"];for(std::string name:{"saved","restored","evicted"})metrics_["flashengine_prefix_state_"+name+"_total"]=state.value(name,uint64_t(0));metrics_["flashengine_prefix_state_bytes"]=state.value("bytes",uint64_t(0));metrics_["flashengine_prefix_state_entries"]=state.value("entries",uint64_t(0));}
  for(std::string kind:{"decode","draft","verify"})metrics_["flashengine_"+kind+"_graph_replays_total"]=runtime.value(kind+"_graph_replays",uint64_t(0));
  auto info=config_.json();info["engine_version"]="0.4.0";info["contract_version"]=contract_version;info["maturity"]="candidate";info["implementation_revision"]="backend-parity-source-2026-10-17";info["implementation_status"]="unvalidated-source-changes";info["runtime"]=runtime;
  snapshot_={{"info",info},{"metrics",metrics_},{"requests",recent_}};
}
void Engine::emit(const std::shared_ptr<State>& state){
  auto& s=*state->seq;size_t generated=s.tokens.size()-s.prompt_size;
  if(generated==state->emitted&&s.status!=Status::finished)return;
  OutputEvent event;event.request_id=s.id;event.finished=s.status==Status::finished;event.finish_reason=s.finish_reason;
  event.token_ids.assign(s.tokens.begin()+s.prompt_size+state->emitted,s.tokens.end());
  if(!event.token_ids.empty()&&state->ttft<0){require(state->started.has_value(),"输出早于首次调度");state->ttft=std::chrono::duration<double>(Clock::now()-*state->started).count();metrics_["flashengine_ttft_seconds_sum"]=metrics_["flashengine_ttft_seconds_sum"].get<double>()+state->ttft;metrics_["flashengine_ttft_seconds_count"]=metrics_["flashengine_ttft_seconds_count"].get<uint64_t>()+1;}
  metrics_["flashengine_generated_tokens_total"]=metrics_["flashengine_generated_tokens_total"].get<uint64_t>()+event.token_ids.size();state->emitted=generated;
  if(event.finished){metrics_["flashengine_requests_total"]=metrics_["flashengine_requests_total"].get<uint64_t>()+1;if(s.finish_reason=="cancelled")metrics_["flashengine_cancelled_requests_total"]=metrics_["flashengine_cancelled_requests_total"].get<uint64_t>()+1;double queue=state->started?std::chrono::duration<double>(*state->started-state->submitted).count():std::chrono::duration<double>(Clock::now()-state->submitted).count();recent_.push_back({{"request_id",s.id},{"in",s.prompt_size},{"out",generated},{"ttft",state->ttft},{"queue_seconds",queue},{"preemptions",s.preemptions},{"finish",s.finish_reason}});if(recent_.size()>200)recent_.erase(recent_.begin());--inflight_;std::cerr<<"[DONE ] rid="<<s.id<<" in="<<s.prompt_size<<" out="<<generated<<" ttft="<<state->ttft<<" queue="<<queue<<" preempt="<<s.preemptions<<" finish="<<s.finish_reason<<'\n';}
  state->sink(std::move(event));
}
void Engine::loop(){
  try{
    for(;;){
      {std::unique_lock lock(mutex_);cv_.wait(lock,[&]{return stopping_||!pending_.empty()||!scheduler_.empty();});while(!pending_.empty()){auto state=pending_.front();pending_.pop_front();scheduler_.add(state->seq);states_[state->seq->id]=state;}if(stopping_)for(auto& [id,state]:states_)state->seq->cancelled->store(true);}
      auto cancelled=scheduler_.reap_cancelled();for(auto& seq:cancelled){emit(states_.at(seq->id));states_.erase(seq->id);}
      if(scheduler_.empty()){std::lock_guard lock(mutex_);refresh();if(stopping_&&pending_.empty())break;continue;}
      auto batch=scheduler_.schedule();if(batch.sequences.empty())continue;auto start=Clock::now();for(auto& seq:batch.sequences){auto& state=states_.at(seq->id);if(!state->started){state->started=start;std::cerr<<"[REQ  ] rid="<<seq->id<<" wait="<<std::chrono::duration<double>(start-state->submitted).count()<<'\n';}}auto result=model_->run(batch);double seconds=std::chrono::duration<double>(Clock::now()-start).count();
      metrics_["flashengine_engine_steps_total"]=metrics_["flashengine_engine_steps_total"].get<uint64_t>()+1;metrics_["flashengine_engine_busy_seconds_total"]=metrics_["flashengine_engine_busy_seconds_total"].get<double>()+seconds;
      auto key=batch.prefill?"flashengine_prefill_steps_total":"flashengine_decode_steps_total";metrics_[key]=metrics_[key].get<uint64_t>()+1;
      for(auto& round:result.rounds){metrics_["flashengine_spec_rounds_total"]=metrics_["flashengine_spec_rounds_total"].get<uint64_t>()+1;metrics_["flashengine_spec_committed_tokens_total"]=metrics_["flashengine_spec_committed_tokens_total"].get<uint64_t>()+round.at("committed_ids").size();auto proposed=round.at("draft_ids").size(),accepted=round.at("accepted_count").get<size_t>();require(accepted<=proposed,"投机接受计数非法");for(size_t p=0;p<proposed;++p){auto key="flashengine_spec_position_"+std::to_string(p+1);metrics_[key+"_proposed_total"]=metrics_.value(key+"_proposed_total",uint64_t(0))+1;metrics_[key+"_accepted_total"]=metrics_.value(key+"_accepted_total",uint64_t(0))+(p<accepted?1:0);}metrics_["flashengine_spec_draft_tokens_total"]=metrics_.value("flashengine_spec_draft_tokens_total",uint64_t(0))+proposed;metrics_["flashengine_spec_accepted_tokens_total"]=metrics_.value("flashengine_spec_accepted_tokens_total",uint64_t(0))+accepted;}
      for(size_t i=0;i<batch.sequences.size();++i){auto s=batch.sequences[i];if(batch.prefill)s->cached+=s->scheduled;scheduler_.commit(s,result.tokens.at(i));if(s->status!=Status::finished&&!result.tokens[i].empty())s->cached=s->tokens.size()-1;s->scheduled=0;emit(states_.at(s->id));if(s->status==Status::finished)states_.erase(s->id);}
      {std::lock_guard lock(mutex_);refresh();}
    }
  }catch(const std::exception& error){std::cerr<<"[FATAL] 引擎异常："<<error.what()<<std::endl;std::_Exit(70);}
}
}
