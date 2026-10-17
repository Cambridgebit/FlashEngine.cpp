#pragma once
#include "fe/bundle.hpp"
#include "fe/scheduler.hpp"
#include "fe/tokenizer.hpp"

namespace fe {
// 目标 Prefix 仍按 token 匹配；附属快照必须绑定其实际物理末页，防止重算/复用关联旧状态。
inline std::vector<int64_t> prefix_snapshot_key(const std::vector<int64_t>& tokens,size_t end,int32_t page){require(page>=0&&end>0&&end<=tokens.size(),"附属 Prefix key 越界");std::vector<int64_t> key(tokens.begin(),tokens.begin()+end);key.push_back(-int64_t(page)-1);return key;}
struct ModelOutput {std::vector<std::vector<int64_t>> tokens;Json rounds=Json::array(),samples=Json::array();};
class DeviceModel {
public:
  virtual ~DeviceModel()=default;
  virtual ModelOutput run(const Batch& batch)=0;
  virtual void drop(uint64_t id)=0;
  virtual Json snapshot()const=0;
  // 主机查表：状态型模型只有在附属快照完整时才允许复用目标 KV。
  virtual bool prefix_ready(const std::vector<int64_t>&)const{return true;}
  virtual void evict_prefix(const std::vector<int64_t>&){}
  virtual bool prefix_page_ready(const std::vector<int64_t>& key,int32_t)const{return prefix_ready(key);}
  virtual void evict_prefix_page(const std::vector<int64_t>& key,int32_t){evict_prefix(key);}
};
inline void bind_model(Scheduler& scheduler,DeviceModel& model){
  scheduler.on_release=[&model](uint64_t id){model.drop(id);};
  scheduler.blocks.prefix_page_ready=[&model](const std::vector<int64_t>& key,int32_t page){return model.prefix_page_ready(key,page);};
  scheduler.blocks.on_evict_page=[&model](const std::vector<int64_t>& key,int32_t page){model.evict_prefix_page(key,page);};
}
std::unique_ptr<DeviceModel> create_cuda_model(const Bundle& bundle,const Config& config);
std::unique_ptr<DeviceModel> create_hybrid_model(const Bundle& bundle,const Config& config);
using DeviceFactory=std::function<std::unique_ptr<DeviceModel>(const Bundle&,const Config&)>;
int run_device_cli(const std::string& command,Bundle& bundle,Tokenizer& tokenizer,int argc,char** argv,const std::string& backend,const DeviceFactory& factory,const std::string& default_dtype="bfloat16");
std::unique_ptr<DeviceModel> create_ascend_model(const Bundle& bundle,const Config& config);
}
