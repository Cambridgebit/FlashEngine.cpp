// 纯原生 Ascend 入口：Qwen3 使用 ATB 批量路径，其余组合保留 ACLNN 参考执行。
#include "fe/model.hpp"
#include "ascend_net.hpp"
#include <set>
#include "ascend_qwen3.hpp"

namespace fe { namespace {
using namespace ascend;

struct Entry {
  Program program;T ids,positions,slots,pages,mask,seed,hidden,aux,tokens,logits,top_values,top_ids;
  int rows;bool block;
  Entry(Net& net,const Config& c,int count,bool block_draft,const std::vector<int>& aux_layers):rows(count),block(block_draft){
    ids=program.out({rows},ACL_INT64);positions=program.out({rows},ACL_INT64);slots=program.out({rows},ACL_INT64);pages=program.out({c.max_model_len},ACL_INT64);mask=program.out({rows,c.max_model_len});for(auto& input:{ids,positions,slots,pages,mask})zero(input);if(net.eagle){seed=program.out({rows,net.h});zero(seed);}
    auto output=net.forward(program,ids,positions,slots,pages,mask,aux_layers,seed);hidden=output.first;aux=output.second;auto sampled=net.eagle?net.norm(program,hidden,net.w("norm.weight")):hidden;
    if(block)sampled=program.narrow(sampled,0,1,rows-1);logits=program.linear(sampled,net.head);tokens=program.argmax(logits);
    if(net.eagle){auto mapped=program.view(program.gather(net.w("d2t"),tokens,0),tokens->shape);tokens=program.binary(tokens,mapped,'+');}
    if(c.trace_logits){auto top=program.topk(program.cast(logits,ACL_FLOAT),std::min<int64_t>(8,logits->shape.back()));top_values=top.first;top_ids=top.second;}
    program.finish();
  }
  void prepare(const Seq& s,const Config& c,size_t start,const std::vector<int64_t>& input){
    require(input.size()==size_t(rows)&&start+rows<=size_t(c.max_model_len),"ACL 查询超出上下文");std::vector<int64_t> pos(rows),physical(rows),read(c.max_model_len,c.num_kvcache_blocks*c.kvcache_block_size);std::vector<uint16_t> bias(rows*c.max_model_len);
    for(int j=0;j<rows;++j){pos[j]=start+j;physical[j]=s?int64_t(s->blocks.at(pos[j]/c.kvcache_block_size))*c.kvcache_block_size+pos[j]%c.kvcache_block_size:c.num_kvcache_blocks*c.kvcache_block_size+j;for(int t=0;t<c.max_model_len;++t)bias[j*c.max_model_len+t]=fp16(t<int(block?start+rows:start+j+1)?0.f:-INFINITY);}
    for(size_t t=0;t<start+rows;++t)read[t]=s?int64_t(s->blocks.at(t/c.kvcache_block_size))*c.kvcache_block_size+t%c.kvcache_block_size:c.num_kvcache_blocks*c.kvcache_block_size+t%rows;
    upload(ids,input.data(),input.size()*8);upload(positions,pos.data(),pos.size()*8);upload(slots,physical.data(),physical.size()*8);upload(pages,read.data(),read.size()*8);upload(mask,bias.data(),bias.size()*2);
  }
  void capture(const Config& c,aclrtStream stream){prepare({},c,0,std::vector<int64_t>(rows,0));if(seed)zero(seed);program.capture(stream);}
};
struct Projection {
  Program program;T aux,positions,slots,seed;bool eagle;
  Projection(Net& net,int width):eagle(net.eagle){aux=program.out({1,width});zero(aux);if(eagle)seed=program.linear(aux,net.w("fc.weight"));else{positions=program.out({1},ACL_INT64);slots=program.out({1},ACL_INT64);zero(positions);zero(slots);net.project(program,aux,positions,slots);}program.finish();}
};
struct RequestState {
  int slot=0;bool cold=true;int64_t draft_token=0;HostTensor auxiliary,carry;
  std::vector<uint8_t> pending;size_t pending_begin=0,pending_filled=0;
};
class AscendModel final:public DeviceModel {
  Runtime runtime_;Config config_,draft_config_;std::unique_ptr<Net> target_,draft_;std::unique_ptr<Entry> target_row_,draft_row_,draft_block_;std::unique_ptr<Projection> projection_;
  std::vector<int> aux_layers_;std::map<uint64_t,RequestState> requests_;std::set<int> free_;PrefixStates prefix_;int k_=0;bool eagle_=false;
  uint64_t decode_=0,draft_replays_=0,verify_=0,eager_=0,projected_=0,rebuilt_=0;size_t private_pages_=0;
  Seq draft_view(const Seq& s){if(!eagle_)return s;auto view=std::make_shared<Sequence>(*s);view->blocks.clear();auto slot=requests_.at(s->id).slot;for(size_t j=0;j<private_pages_;++j)view->blocks.push_back(slot*private_pages_+j);return view;}
  void select(const Seq& s){int64_t slot=requests_.at(s->id).slot;upload(target_->state_slot,&slot,8);}
  HostTensor project_seed(const HostTensor& auxiliary){restore(projection_->aux,auxiliary);projection_->program.run(runtime_.stream);return download(projection_->seed,runtime_.stream);}
  void project_context(const Seq& s,size_t position,const HostTensor& auxiliary){
    restore(projection_->aux,auxiliary);int64_t pos=position,slot=int64_t(s->blocks.at(position/config_.kvcache_block_size))*config_.kvcache_block_size+position%config_.kvcache_block_size;upload(projection_->positions,&pos,8);upload(projection_->slots,&slot,8);projection_->program.run(runtime_.stream);check(aclrtSynchronizeStream(runtime_.stream),"提交 DFlash 上下文投影");++projected_;
  }
  std::pair<int64_t,HostTensor> eagle_step(const Seq& s,size_t position,int64_t input,const HostTensor& seed,bool graph){
    draft_row_->prepare(draft_view(s),draft_config_,position,{input});restore(draft_row_->seed,seed);execute(*draft_row_,graph,"draft");auto id=read_ids(draft_row_->tokens,runtime_.stream).at(0);return {id,download(draft_row_->hidden,runtime_.stream)};
  }
  void record_eagle(const Seq& s,const std::vector<int64_t>& path,size_t pos,const HostTensor& aux){
    if(!config_.enable_prefix_cache)return;auto& state=requests_.at(s->id);size_t block=config_.kvcache_block_size,offset=pos%block,row=aux.data.size();require(aux.shape[0]==1,"EAGLE aux 必须为单行");
    if(!offset){state.pending.assign(block*row,0);state.pending_begin=pos;state.pending_filled=0;}
    require(state.pending_begin+state.pending_filled==pos&&state.pending.size()==block*row,"EAGLE aux 前缀不连续");std::copy(aux.data.begin(),aux.data.end(),state.pending.begin()+offset*row);++state.pending_filled;
    if(state.pending_filled==block){require(pos+1<=path.size(),"EAGLE prefix key 越界");HostTensor page=aux;page.shape[0]=block;page.data=std::move(state.pending);prefix_.put(prefix_snapshot_key(path,pos+1,s->blocks.at(pos/block)),{std::move(page)});state.pending_filled=0;}
  }
  void save_prefix(const Seq& s,const std::vector<int64_t>& path,size_t end,const HostState& state){if(config_.enable_prefix_cache&&end%config_.kvcache_block_size==0){require(end<=path.size(),"GDN prefix key 越界");prefix_.put(prefix_snapshot_key(path,end,s->blocks.at((end-1)/config_.kvcache_block_size)),state);}}
  void acquire(const Seq& s){
    if(requests_.contains(s->id))return;require(!free_.empty(),"ACL 请求状态槽耗尽");int slot=*free_.begin();free_.erase(slot);RequestState state;state.slot=slot;requests_.emplace(s->id,std::move(state));target_->clear(slot);
    if(!s->cached)return;
    if(target_->recurrent()){auto key=prefix_snapshot_key(s->tokens,s->cached,s->blocks.at((s->cached-1)/config_.kvcache_block_size));require(prefix_.has(key),"GDN 目标 Prefix 命中但缺少状态");target_->load(slot,prefix_.get(key));}
    if(eagle_){size_t block=config_.kvcache_block_size;for(size_t begin=0;begin<s->cached;begin+=block){auto key=prefix_snapshot_key(s->tokens,begin+block,s->blocks.at((begin+block-1)/block));auto page=prefix_.get(key).at(0);size_t width=page.data.size()/block;
        for(size_t j=0;j<block;++j){HostTensor row=page;row.shape[0]=1;row.data.assign(page.data.begin()+j*width,page.data.begin()+(j+1)*width);eagle_step(s,begin+j,s->tokens.at(begin+j+1),project_seed(row),!config_.enforce_eager);++rebuilt_;}}
    }
  }
  void execute(Entry& entry,bool graph,const std::string& kind){
    if(graph&&entry.program.graph){check(aclmdlRIExecuteAsync(entry.program.graph,runtime_.stream),"ACL Graph replay");if(kind=="draft")++draft_replays_;else if(kind=="verify")++verify_;else ++decode_;}else{entry.program.eager(runtime_.stream);++eager_;}
  }
  int64_t target_step(const Seq& s,size_t pos,int64_t input,bool graph,bool verify=false){select(s);target_row_->prepare(s,config_,pos,{input});execute(*target_row_,graph,verify?"verify":"decode");return read_ids(target_row_->tokens,runtime_.stream).at(0);}
  Json sample(const Seq& s){auto ids=read_ids(target_row_->top_ids,runtime_.stream);auto host=download(target_row_->top_values,runtime_.stream);std::vector<float> values(ids.size());std::memcpy(values.data(),host.data.data(),host.data.size());return {{"token_index",s->tokens.size()-s->prompt_size},{"ids",ids},{"values",values}};}
  ModelOutput standard(const Batch& batch){
    ModelOutput out;for(auto& s:batch.sequences){size_t begin=batch.prefill?s->cached:s->tokens.size()-1,count=batch.prefill?s->scheduled:1;int64_t next=0;
      for(size_t pos=begin;pos<begin+count;++pos){next=target_step(s,pos,s->tokens.at(pos),!batch.prefill&&!config_.enforce_eager);HostTensor aux;if(draft_)aux=download(target_row_->aux,runtime_.stream);
        if(eagle_){record_eagle(s,s->tokens,pos,aux);auto predicted=eagle_step(s,pos,pos+1<s->tokens.size()?s->tokens[pos+1]:next,project_seed(aux),false);auto& state=requests_.at(s->id);state.auxiliary=aux;state.draft_token=predicted.first;state.cold=true;state.carry={};}
        else if(draft_)project_context(s,pos,aux);
        if(target_->recurrent()&&config_.enable_prefix_cache&&(pos+1)%config_.kvcache_block_size==0)save_prefix(s,s->tokens,pos+1,target_->save(requests_.at(s->id).slot,runtime_.stream));
      }
      bool complete=!batch.prefill||begin+count==s->tokens.size();out.tokens.push_back(complete?std::vector<int64_t>{next}:std::vector<int64_t>{});if(complete&&config_.trace_logits)out.samples.push_back(sample(s));
    }return out;
  }
  ModelOutput dflash(const Seq& s){
    size_t start=s->tokens.size()-1;std::vector<int64_t> input(k_+1,draft_->config.at("mask_token_id"));input[0]=s->tokens.back();draft_block_->prepare(s,config_,start,input);execute(*draft_block_,!config_.enforce_eager,"draft");auto proposed=read_ids(draft_block_->tokens,runtime_.stream);std::vector<int64_t> verified;std::vector<HostTensor> auxiliary;std::vector<HostState> states;
    for(int j=0;j<=k_;++j){verified.push_back(target_step(s,start+j,j?proposed.at(j-1):s->tokens.back(),!config_.enforce_eager,true));auxiliary.push_back(download(target_row_->aux,runtime_.stream));if(target_->recurrent())states.push_back(target_->save(requests_.at(s->id).slot,runtime_.stream));}
    auto committed=accept_greedy(proposed,verified);size_t count=committed.size();if(target_->recurrent())target_->load(requests_.at(s->id).slot,states.at(count-1));auto path=s->tokens;path.insert(path.end(),proposed.begin(),proposed.begin()+count-1);
    for(size_t j=0;j<count;++j){project_context(s,start+j,auxiliary[j]);if(target_->recurrent())save_prefix(s,path,start+j+1,states[j]);}
    return round(s,proposed,verified,committed,start);
  }
  ModelOutput eagle(const Seq& s){
    auto& state=requests_.at(s->id);size_t start=s->tokens.size()-1;auto seed=state.cold?project_seed(state.auxiliary):state.carry;std::vector<int64_t> proposed{state.draft_token};
    for(int j=1;j<k_;++j){auto value=eagle_step(s,start+j-1,proposed.back(),seed,!config_.enforce_eager);proposed.push_back(value.first);seed=std::move(value.second);}
    std::vector<int64_t> verified;std::vector<HostTensor> auxiliary;for(int j=0;j<=k_;++j){verified.push_back(target_step(s,start+j,j?proposed.at(j-1):s->tokens.back(),!config_.enforce_eager,true));auxiliary.push_back(download(target_row_->aux,runtime_.stream));}
    auto committed=accept_greedy(proposed,verified);size_t m=committed.size()-1;for(int j=0;j<=k_;++j){int64_t shifted=j<k_?verified[j+1]:verified[k_];if(size_t(j)==m)shifted=verified[m];auto value=eagle_step(s,start+j,shifted,project_seed(auxiliary[j]),!config_.enforce_eager);if(size_t(j)==m){state.carry=std::move(value.second);state.draft_token=value.first;state.auxiliary=auxiliary[j];}}
    state.cold=false;auto path=s->tokens;path.insert(path.end(),proposed.begin(),proposed.begin()+m);for(size_t j=0;j<=m;++j)record_eagle(s,path,start+j,auxiliary[j]);return round(s,proposed,verified,committed,start);
  }
  ModelOutput round(const Seq& s,const std::vector<int64_t>& proposed,const std::vector<int64_t>& verified,const std::vector<int64_t>& committed,size_t start){ModelOutput out;out.tokens.push_back(committed);out.rounds.push_back({{"request_id",s->id},{"draft_ids",proposed},{"verify_ids",verified},{"accepted_count",committed.size()-1},{"committed_ids",committed},{"context_length",start}});return out;}
public:
  AscendModel(const Bundle& bundle,Config config):config_(config),draft_config_(config),prefix_(config.enable_prefix_cache?config.recurrent_prefix_cache_bytes:0){
    require(config.dtype=="float16","310P 参考执行仅支持 float16，GDN SSM 保持 float32");require(config.max_model_len<=bundle.config["target"]["max_position_embeddings"].get<int64_t>(),"模型位置上限不足");target_=std::make_unique<Net>(bundle.target_weights,bundle.config.at("target"),config);
    for(int slot=0;slot<config.max_num_seqs;++slot)free_.insert(slot);
    if(!bundle.config.at("draft").is_null()){auto dc=bundle.config.at("draft");eagle_=dc.value("spec_method",std::string("dflash"))=="eagle3";require(!target_->hybrid||!eagle_,"Qwen3.5 不支持 EAGLE3");k_=dc.at("num_tokens");require(k_>0&&k_+1<=config.kvcache_block_size,"草稿长度超出 dummy 页");
      if(eagle_){private_pages_=(config.max_model_len+config.kvcache_block_size-1)/config.kvcache_block_size;draft_config_.num_kvcache_blocks=private_pages_*config.max_num_seqs;draft_config_.validate();aux_layers_=dc.at("aux_layers").get<std::vector<int>>();}else for(int id:dc.at("target_layer_ids").get<std::vector<int>>())aux_layers_.push_back(id+1);
      draft_=std::make_unique<Net>(bundle.draft_weights,dc,draft_config_,target_->embed,target_->head);projection_=std::make_unique<Projection>(*draft_,target_->h*aux_layers_.size());
    }
    startup_trace("开始准备 target 单行执行计划");target_row_=std::make_unique<Entry>(*target_,config_,1,false,aux_layers_);startup_trace("target 单行执行计划就绪");if(!config.enforce_eager){startup_trace("开始捕获 target Graph");target_row_->capture(config_,runtime_.stream);startup_trace("target Graph 捕获完成");}
    if(draft_){startup_trace("开始准备 draft 执行计划");auto entry=std::make_unique<Entry>(*draft_,draft_config_,eagle_?1:k_+1,!eagle_,std::vector<int>{});if(!config.enforce_eager)entry->capture(draft_config_,runtime_.stream);if(eagle_)draft_row_=std::move(entry);else draft_block_=std::move(entry);startup_trace("draft 执行计划就绪");}
  }
  ~AscendModel()override{runtime_.enter();aclrtSynchronizeStream(runtime_.stream);}
  ModelOutput run(const Batch& batch)override{
    runtime_.enter();for(auto& s:batch.sequences)acquire(s);if(batch.prefill||!draft_)return standard(batch);ModelOutput out;
    for(auto& s:batch.sequences){ModelOutput value;if(s->tokens.size()+k_+1>size_t(config_.max_model_len))value=standard(Batch{false,{s}});else value=eagle_?eagle(s):dflash(s);out.tokens.push_back(std::move(value.tokens[0]));for(auto& r:value.rounds)out.rounds.push_back(std::move(r));for(auto& v:value.samples)out.samples.push_back(std::move(v));}return out;
  }
  void drop(uint64_t id)override{if(auto i=requests_.find(id);i!=requests_.end()){free_.insert(i->second.slot);requests_.erase(i);}}
  bool prefix_page_ready(const std::vector<int64_t>& key,int32_t page)const override{return (!eagle_&&!target_->recurrent())||prefix_.has(prefix_snapshot_key(key,key.size(),page));}
  void evict_prefix_page(const std::vector<int64_t>& key,int32_t page)override{prefix_.erase(prefix_snapshot_key(key,key.size(),page));}
  Json snapshot()const override{return {{"backend","ascend310p"},{"attention","aclnn-nd-reference"},{"maturity","source-prepared-unvalidated"},{"target_profile",target_->config.at("model_type")},{"spec_method",eagle_?"eagle3":draft_?"dflash":"none"},{"quantization",target_->config.value("quantization",std::string("none"))},{"quantized_compute",false},{"decode_graph_replays",decode_},{"draft_graph_replays",draft_replays_},{"verify_graph_replays",verify_},{"eager_rounds",eager_},{"draft_projected_tokens",projected_},{"eagle_prefix_rebuilt_tokens",rebuilt_},{"eagle_active_requests",eagle_?requests_.size():0},{"eagle_kv_active_requests",eagle_?requests_.size():0},{"recurrent_active_requests",target_->recurrent()?requests_.size():0},{"prefix_state",prefix_.snapshot()},{"graph_buckets",config_.enforce_eager?std::vector<int>{}:std::vector<int>{1}},{"graph_execution","sequential-single-row"}};}
};
}
std::unique_ptr<DeviceModel> create_ascend_model(const Bundle& bundle,const Config& config){
  const auto& target=bundle.config.at("target");const auto& draft=bundle.config.at("draft");
  bool block=draft.is_null()||draft.value("spec_method",std::string("dflash"))=="dflash";
  if(target.at("model_type")=="qwen3"&&block)return std::make_unique<ascend::QwenAscendModel>(bundle,config);
  return std::make_unique<AscendModel>(bundle,config);
}
}
