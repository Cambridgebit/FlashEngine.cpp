#pragma once
#include "fe/common.hpp"
#include <ATen/ATen.h>
#include <list>
#include <map>

namespace fe {
// 有界 CPU 快照缓存；key 是完整 token 前缀及末页绑定。snapshot()/has() 不访问设备。
class PrefixStateCache {
  using Key=std::vector<int64_t>;
  struct Entry {std::vector<at::Tensor> tensors;size_t bytes;std::list<Key>::iterator lru;};
  size_t limit_,bytes_=0;std::list<Key> lru_;std::map<Key,Entry> entries_;
  uint64_t saved_=0,restored_=0,evicted_=0;
public:
  explicit PrefixStateCache(size_t limit):limit_(limit){}
  bool has(const Key& key)const{return entries_.contains(key);}
  void erase(const Key& key){if(auto it=entries_.find(key);it!=entries_.end()){bytes_-=it->second.bytes;lru_.erase(it->second.lru);entries_.erase(it);++evicted_;}}
  void put(const Key& key,const std::vector<at::Tensor>& values){
    // 物理页重算后同绑定必须刷新；已发布共享页本身不得被写入。
    erase(key);
    size_t bytes=0;for(auto& t:values)bytes+=t.numel()*t.element_size();if(bytes>limit_||!limit_)return;
    while(bytes_+bytes>limit_)erase(lru_.front());
    std::vector<at::Tensor> host;for(auto& t:values)host.push_back(t.to(at::kCPU).clone());
    lru_.push_back(key);entries_.emplace(key,Entry{std::move(host),bytes,std::prev(lru_.end())});bytes_+=bytes;++saved_;
  }
  const std::vector<at::Tensor>& get(const Key& key){auto& e=entries_.at(key);lru_.splice(lru_.end(),lru_,e.lru);++restored_;return e.tensors;}
  Json snapshot()const{return {{"saved",saved_},{"restored",restored_},{"evicted",evicted_},{"bytes",bytes_},{"entries",entries_.size()},{"budget_bytes",limit_}};}
};
}
