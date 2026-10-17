#include "fe/scheduler.hpp"
#include <algorithm>

namespace fe {
BlockManager::BlockManager(int64_t total,int64_t block_size,bool caching):size_(block_size),total_(total),caching_(caching){require(total>0&&total<INT32_MAX&&block_size>0,"非法 KV 池");refs_.resize(total);for(int32_t i=0;i<total;++i)free_.push_back(i);}
void BlockManager::probe(Sequence& s){
  if(!caching_||s.prefix_probed)return;
  require(s.blocks.empty()&&s.cached==0,"前缀探测必须早于分配");s.prefix_probed=true;
  if(!s.prefix_counted){++requests_;s.prefix_counted=true;}
  // 保留至少一个输入 token 重算 logits；共享页永不写入。
  for(size_t end=size_;end<s.tokens.size();end+=size_){
    std::vector<int64_t> key(s.tokens.begin(),s.tokens.begin()+end);auto it=cached_.find(key);if(it==cached_.end()||(prefix_ready&&!prefix_ready(key))||(prefix_page_ready&&!prefix_page_ready(key,it->second)))break;
    auto id=it->second;if(refs_[id]++==0){std::erase(free_,id);require(used_.insert(id).second,"缓存页引用异常");}
    s.blocks.push_back(id);s.cached=end;++hit_blocks_;
  }
  if(s.cached&&!s.prefix_hit_counted){++hits_;s.prefix_hit_counted=true;}
}
void BlockManager::publish(const Sequence& s,size_t computed){
  if(!caching_)return;
  computed=std::min(computed,s.tokens.size());
  for(size_t end=size_;end<=computed;end+=size_){
    auto page=(end-1)/size_;require(page<s.blocks.size(),"发布未分配的 KV 页");auto id=s.blocks[page];
    std::vector<int64_t> key(s.tokens.begin(),s.tokens.begin()+end);
    if(prefix_ready&&!prefix_ready(key))continue;
    if(prefix_page_ready&&!prefix_page_ready(key,id))continue;
    if(cached_.contains(key))continue;
    require(!keys_.contains(id),"不得修改已发布的不可变 KV 页");cached_.emplace(key,id);keys_.emplace(id,std::move(key));++new_blocks_;
  }
}
Json BlockManager::snapshot()const{return {{"prefix_requests",requests_},{"prefix_hit_requests",hits_},{"prefix_cached_blocks",hit_blocks_},{"prefix_new_blocks",new_blocks_},{"prefix_resident_blocks",cached_.size()},{"prefix_evictions",evictions_}};}
bool BlockManager::ensure(Sequence& s,size_t target){
  size_t needed=(target+size_-1)/size_;if(needed<=s.blocks.size())return true;
  if(needed-s.blocks.size()>free_.size())return false;
  while(s.blocks.size()<needed){auto id=free_.front();free_.pop_front();if(auto old=keys_.find(id);old!=keys_.end()){if(on_evict)on_evict(old->second);if(on_evict_page)on_evict_page(old->second,id);cached_.erase(old->second);keys_.erase(old);++evictions_;}require(refs_[id]==0&&used_.insert(id).second,"页重复分配");refs_[id]=1;s.blocks.push_back(id);}return true;
}
void BlockManager::release(Sequence& s){for(auto id:s.blocks){require(refs_[id]>0,"重复释放 KV 页");if(--refs_[id]==0){require(used_.erase(id)==1,"页引用不一致");free_.push_back(id);}}s.blocks.clear();s.cached=0;s.scheduled=0;s.prefix_probed=false;}
Scheduler::Scheduler(Config c):config_(c),blocks(c.num_kvcache_blocks,c.kvcache_block_size,c.enable_prefix_cache){config_.validate();}
size_t Scheduler::active()const{return running_.size()+std::count_if(waiting_.begin(),waiting_.end(),[](const Seq& s){return !s->blocks.empty();});}
void Scheduler::add(Seq s){
  if(s->tokens.empty()||s->max_tokens<=0||s->tokens.size()>=size_t(config_.max_model_len))throw InputError("prompt/输出长度非法");
  if((s->tokens.size()+config_.kvcache_block_size-1)/config_.kvcache_block_size>size_t(blocks.total()))throw InputError("prompt 超过 KV 池容量");
  waiting_.push_back(std::move(s));
}
void Scheduler::preempt(const Seq& s){
  std::erase(running_,s);blocks.release(*s);if(on_release)on_release(s->id);s->status=Status::waiting;waiting_.push_back(s);++preemptions;++s->preemptions;
}
Batch Scheduler::prefill(){
  Batch batch{true,{}};int64_t budget=config_.max_num_batched_tokens;
  size_t partial=std::count_if(waiting_.begin(),waiting_.end(),[](const Seq& s){return !s->blocks.empty();});
  for(auto it=waiting_.begin();it!=waiting_.end()&&budget>0&&batch.sequences.size()<size_t(config_.max_num_seqs);){
    auto s=*it;bool existing=!s->blocks.empty();if(!existing&&active()>=size_t(config_.max_num_seqs))break;
    blocks.probe(*s);
    size_t remaining=s->tokens.size()-s->cached;bool will_partial=remaining>size_t(budget);
    if(will_partial&&(!config_.enable_chunked_prefill||(!existing&&partial>=size_t(config_.max_num_partial_prefills)))){if(!existing)blocks.release(*s);break;}
    if(!blocks.ensure(*s,s->tokens.size())){
      // Prefix 探测会持有共享页；没有真正准入的新请求不得跨步占住这些引用。
      if(!existing)blocks.release(*s);
      if(!batch.sequences.empty()||running_.empty())break;
      auto victim=*std::min_element(running_.begin(),running_.end(),[](const Seq& a,const Seq& b){return a->touched<b->touched;});
      preempt(victim);it=waiting_.begin();continue;
    }
    if(!existing)++admitted_;
    s->scheduled=std::min(remaining,size_t(budget));require(s->scheduled>0,"prefill 无进展");budget-=s->scheduled;s->touched=++clock_;batch.sequences.push_back(s);
    if(s->cached+s->scheduled==s->tokens.size()){s->status=Status::running;running_.push_back(s);it=waiting_.erase(it);if(existing)--partial;}
    else{if(!existing)++partial;break;}
  }
  if(!batch.sequences.empty()){++steps_;++prefill_steps_;bool chunked=false;for(auto& s:batch.sequences){prefill_tokens_+=s->scheduled;chunked|=s->cached+s->scheduled<s->tokens.size();}if(chunked)++chunked_steps_;}
  return batch;
}
Batch Scheduler::decode(){
  Batch batch;auto pending=running_;
  for(auto& s:pending){
    if(s->status!=Status::running)continue;
    if(s->tokens.size()>=size_t(config_.max_model_len)){finish(s,"length");continue;}
    const auto reserve=std::min(size_t(config_.max_model_len),s->tokens.size()+size_t(config_.spec_num_tokens ? config_.spec_num_tokens+1 : 0));
    while(!blocks.ensure(*s,reserve)){
      Seq victim;for(auto& candidate:running_)if(candidate!=s&&std::find(batch.sequences.begin(),batch.sequences.end(),candidate)==batch.sequences.end()&&(!victim||candidate->touched<victim->touched))victim=candidate;
      if(victim)preempt(victim);else{preempt(s);break;}
    }
    if(s->status!=Status::running)continue;
    s->scheduled=1;s->touched=++clock_;batch.sequences.push_back(s);
    if(batch.sequences.size()>=size_t(config_.max_num_seqs))break;
  }
  if(!batch.sequences.empty()){++steps_;++decode_steps_;decode_tokens_+=batch.sequences.size();}
  return batch;
}
Batch Scheduler::schedule(){
  if(!waiting_.empty()&&(!prefer_decode_||running_.empty())){auto b=prefill();if(!b.sequences.empty()){prefer_decode_=true;return b;}}
  auto b=decode();if(!b.sequences.empty()){prefer_decode_=false;return b;}
  b=prefill();if(!b.sequences.empty()){prefer_decode_=true;return b;}
  require(empty(),"调度无进展：KV 容量不足");return {};
}
void Scheduler::finish(const Seq& s,std::string reason){if(s->status==Status::finished)return;s->status=Status::finished;s->finish_reason=std::move(reason);std::erase(waiting_,s);std::erase(running_,s);blocks.release(*s);if(on_release)on_release(s->id);}
void Scheduler::commit(const Seq& s,const std::vector<int64_t>& tokens){
  if(s->status==Status::finished)return;
  blocks.publish(*s,s->status==Status::running?s->tokens.size():s->cached);
  for(auto token:tokens){
    if(s->tokens.size()>=size_t(config_.max_model_len)){finish(s,"length");break;}
    s->tokens.push_back(token);
    if(!s->ignore_eos&&token==config_.eos){finish(s,"stop");break;}
    if(s->tokens.size()-s->prompt_size>=size_t(s->max_tokens)||s->tokens.size()>=size_t(config_.max_model_len)){finish(s,"length");break;}
  }
}
std::vector<Seq> Scheduler::reap_cancelled(){
  std::vector<Seq> dead;for(auto& s:waiting_)if(s->cancelled->load())dead.push_back(s);for(auto& s:running_)if(s->cancelled->load())dead.push_back(s);
  for(auto& s:dead)finish(s,"cancelled");
  return dead;
}
Json Scheduler::snapshot()const{auto out=blocks.snapshot();out.update(Json{{"running",running_.size()},{"waiting",waiting_.size()},{"partial_prefills",active()-running_.size()},{"active",active()},{"preemptions",preemptions},{"used_blocks",blocks.used()},{"total_blocks",blocks.total()},{"steps",steps_},{"prefill_steps",prefill_steps_},{"decode_steps",decode_steps_},{"prefill_tokens",prefill_tokens_},{"decode_tokens",decode_tokens_},{"chunked_prefill_steps",chunked_steps_},{"admitted_requests",admitted_}});return out;}
std::vector<int64_t> accept_greedy(const std::vector<int64_t>& draft,const std::vector<int64_t>& verify){require(verify.size()==draft.size()+1,"verify 长度错误");size_t accepted=0;while(accepted<draft.size()&&draft[accepted]==verify[accepted])++accepted;auto result=std::vector<int64_t>(draft.begin(),draft.begin()+accepted);result.push_back(verify[accepted]);return result;}
}
