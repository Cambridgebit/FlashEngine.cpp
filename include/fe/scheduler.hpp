#pragma once
#include "fe/common.hpp"
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <map>

namespace fe {
enum class Status {waiting,running,finished};
struct Sequence {
  uint64_t id;std::vector<int64_t> tokens;size_t prompt_size,cached=0,scheduled=0;
  int64_t max_tokens;bool ignore_eos=false;Status status=Status::waiting;
  std::vector<int32_t> blocks;uint64_t touched=0,preemptions=0;std::string finish_reason;
  bool prefix_probed=false,prefix_counted=false,prefix_hit_counted=false;
  std::shared_ptr<std::atomic<bool>> cancelled=std::make_shared<std::atomic<bool>>(false);
  Sequence(uint64_t value,std::vector<int64_t> prompt,int64_t limit):id(value),tokens(std::move(prompt)),prompt_size(tokens.size()),max_tokens(limit){}
};
using Seq=std::shared_ptr<Sequence>;
class BlockManager {
  std::deque<int32_t> free_;std::set<int32_t> used_;int64_t size_,total_;
  bool caching_;std::vector<size_t> refs_;
  std::map<std::vector<int64_t>,int32_t> cached_;
  std::map<int32_t,std::vector<int64_t>> keys_;
  uint64_t requests_=0,hits_=0,hit_blocks_=0,new_blocks_=0,evictions_=0;
public:
  std::function<bool(const std::vector<int64_t>&)> prefix_ready;
  std::function<void(const std::vector<int64_t>&)> on_evict;
  std::function<bool(const std::vector<int64_t>&,int32_t)> prefix_page_ready;
  std::function<void(const std::vector<int64_t>&,int32_t)> on_evict_page;
  BlockManager(int64_t total,int64_t block_size,bool caching=false);
  void probe(Sequence& sequence);
  void publish(const Sequence& sequence,size_t computed);
  Json snapshot()const;
  bool ensure(Sequence& sequence,size_t target);
  void release(Sequence& sequence);
  size_t used()const{return used_.size();}
  size_t free()const{return free_.size();}
  int64_t total()const{return total_;}
};
struct Batch {bool prefill=false;std::vector<Seq> sequences;};
class Scheduler {
  Config config_;std::deque<Seq> waiting_,running_;uint64_t clock_=0;bool prefer_decode_=false;
  uint64_t steps_=0,prefill_steps_=0,decode_steps_=0,prefill_tokens_=0,decode_tokens_=0,chunked_steps_=0,admitted_=0;
  Batch prefill();Batch decode();void preempt(const Seq& sequence);size_t active()const;
public:
  BlockManager blocks;uint64_t preemptions=0;
  std::function<void(uint64_t)> on_release;
  explicit Scheduler(Config config);
  void add(Seq sequence);
  Batch schedule();
  void commit(const Seq& sequence,const std::vector<int64_t>& tokens);
  void finish(const Seq& sequence,std::string reason);
  std::vector<Seq> reap_cancelled();
  Json snapshot()const;
  bool empty()const{return waiting_.empty()&&running_.empty();}
};
// 接受处理与设备无关；可用于 native kernel 的对照测试。
std::vector<int64_t> accept_greedy(const std::vector<int64_t>& draft,const std::vector<int64_t>& verify);
}
