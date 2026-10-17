// 无 TorchNPU/ATen 的 AscendCL + ACLNN 参考路径。缺 SDK/真机时只能标记 source-prepared。
#include "fe/model.hpp"
#include "fe/profiles.hpp"
#include "ascend_ops.hpp"
#include <cmath>
#include <iostream>

namespace fe {
namespace {
using namespace ascend;
struct Runtime {
  aclrtContext context=nullptr;aclrtStream stream=nullptr;
  Runtime(){check(aclInit(nullptr),"aclInit");try{check(aclrtSetDevice(0),"aclrtSetDevice");check(aclrtCreateContext(&context,0),"aclrtCreateContext");check(aclrtCreateStream(&stream),"aclrtCreateStream");}catch(...){if(context)aclrtDestroyContext(context);aclrtResetDevice(0);aclFinalize();throw;}}
  ~Runtime(){if(context)aclrtSetCurrentContext(context);if(stream){aclrtSynchronizeStream(stream);aclrtDestroyStream(stream);}if(context)aclrtDestroyContext(context);aclrtResetDevice(0);aclFinalize();}
  void enter(){check(aclrtSetCurrentContext(context),"切换引擎线程 NPU context");}
};
T load_weight(const WeightView& w){auto out=tensor(w.shape);size_t count=elements(w.shape);if(w.dtype=="F16"){upload(out,w.data(),w.bytes);return out;}
  require(w.dtype=="BF16"||w.dtype=="F32","Ascend 只装载浮点权重");std::vector<uint16_t> converted(std::min(count,size_t(1<<19)));auto* source=static_cast<const char*>(w.data());
  for(size_t start=0;start<count;start+=converted.size()){size_t n=std::min(converted.size(),count-start);for(size_t i=0;i<n;++i){float value;if(w.dtype=="BF16"){uint16_t bits;std::memcpy(&bits,source+(start+i)*2,2);value=std::bit_cast<float>(uint32_t(bits)<<16);}else std::memcpy(&value,source+(start+i)*4,4);converted[i]=fp16(value);}check(aclrtMemcpy(static_cast<char*>(out->data())+start*2,(count-start)*2,converted.data(),n*2,ACL_MEMCPY_HOST_TO_DEVICE),"权重转换上传");}return out;
}
struct Net {
  Json config;int h,hq,hkv,dim,layers;double eps;std::map<std::string,T> weights;std::vector<T> keys,values;T embed,head,rope;
  Net(const std::map<std::string,WeightView>& source,Json cfg,const Config& runtime,T shared_embed={},T shared_head={}):config(std::move(cfg)){
    h=config.at("hidden_size");hq=config.at("num_attention_heads");hkv=config.at("num_key_value_heads");dim=config.at("head_dim");layers=config.at("num_hidden_layers");eps=config.at("rms_norm_eps");
    for(auto& [name,w]:source)weights[name]=load_weight(w);
    embed=weights.contains("embed_tokens.weight")?weights.at("embed_tokens.weight"):shared_embed;require(bool(embed),"缺少 Ascend embedding");head=weights.contains("lm_head.weight")?weights.at("lm_head.weight"):shared_head?shared_head:embed;
    // 专家只在加载期聚合，随后释放逐专家副本，避免每个图复制整套权重。
    for(int l=0;l<layers;++l){auto prefix="layers."+std::to_string(l)+".mlp.experts.";if(moe_layer(config,l)&&!weights.contains(prefix+"gate_up_proj.weight")){int e=config.at("num_experts"),m=config.at("moe_intermediate_size");auto gu=tensor({e,2*m,h}),down=tensor({e,h,m});for(int j=0;j<e;++j){auto p=prefix+std::to_string(j)+".";for(auto projection:{std::string("gate"),std::string("up"),std::string("down")}){auto name=p+projection+"_proj.weight";auto value=weights.at(name);auto dest=projection=="down"?down:gu;size_t offset=projection=="down"?size_t(j)*h*m:size_t(j)*2*m*h+(projection=="up"?m*h:0);check(aclrtMemcpy(static_cast<char*>(dest->data())+offset*2,dest->size()-offset*2,value->data(),value->size(),ACL_MEMCPY_DEVICE_TO_DEVICE),"专家聚合");weights.erase(name);}}weights[prefix+"gate_up_proj.weight"]=gu;weights[prefix+"down_proj.weight"]=down;}}
    for(int l=0;l<layers;++l){auto shape=std::vector<int64_t>{(runtime.num_kvcache_blocks+1)*runtime.kvcache_block_size,hkv*dim};keys.push_back(tensor(shape));values.push_back(tensor(shape));check(aclrtMemset(keys.back()->data(),keys.back()->size(),0,keys.back()->size()),"KV 初始化");check(aclrtMemset(values.back()->data(),values.back()->size(),0,values.back()->size()),"KV 初始化");}
    rope=tensor({runtime.max_model_len,dim},ACL_FLOAT);std::vector<float> cache(runtime.max_model_len*dim);double theta=config.at("rope_theta");for(int64_t pos=0;pos<runtime.max_model_len;++pos)for(int j=0;j<dim/2;++j){float frequency=float(pos)/std::pow(float(theta),float(j*2)/dim);cache[pos*dim+j]=std::cos(frequency);cache[pos*dim+j+dim/2]=std::sin(frequency);}upload(rope,cache.data(),cache.size()*4);
  }
  T w(const std::string& name)const{return weights.at(name);}
  T rotate(Program& p,const T& input,const T& positions,int heads){int rows=positions->shape[0];auto x=p.view(p.cast(input,ACL_FLOAT),{rows,heads,dim});auto cs=p.gather(rope,positions,0);auto cos=p.view(p.narrow(cs,1,0,dim/2),{rows,1,dim/2},{dim,dim/2,1}),sin=p.view(p.narrow(cs,1,dim/2,dim/2),{rows,1,dim/2},{dim,dim/2,1});auto a=p.narrow(x,2,0,dim/2),b=p.narrow(x,2,dim/2,dim/2);return p.cast(p.cat({p.binary(p.binary(a,cos,'*'),p.binary(b,sin,'*'),'-'),p.binary(p.binary(b,cos,'*'),p.binary(a,sin,'*'),'+')},2),ACL_FLOAT16);}
  void project(Program& p,const T& aux,const T& positions,const T& slots){int rows=positions->shape[0];auto hidden=p.norm(p.linear(aux,w("fc.weight")),w("hidden_norm.weight"),eps);for(int l=0;l<layers;++l){auto prefix="layers."+std::to_string(l)+".self_attn.";auto k=p.norm(p.view(p.linear(hidden,w(prefix+"k_proj.weight")),{rows,hkv,dim}),w(prefix+"k_norm.weight"),eps);k=rotate(p,k,positions,hkv);auto v=p.linear(hidden,w(prefix+"v_proj.weight"));p.scatter(keys[l],slots,p.view(k,{rows,hkv*dim}));p.scatter(values[l],slots,v);}}
  T mlp(Program& p,const T& hidden,int l){auto prefix="layers."+std::to_string(l)+".mlp.";if(!moe_layer(config,l)){auto gu=p.cat({p.linear(hidden,w(prefix+"gate_proj.weight")),p.linear(hidden,w(prefix+"up_proj.weight"))},1);return p.linear(p.silu_mul(gu),w(prefix+"down_proj.weight"));}
    int n=hidden->shape[0],top=config.at("num_experts_per_tok"),m=config.at("moe_intermediate_size");auto [route,ids]=p.topk(p.softmax(p.linear(hidden,w(prefix+"gate.weight"))),top);if(config.at("norm_topk_prob").get<bool>())route=p.binary(route,p.sum(route,1,true),'/');route=p.cast(route,ACL_FLOAT16);
    std::vector<T> contributions;for(int choice=0;choice<top;++choice){auto selected=p.view(p.narrow(ids,1,choice,1),{n},{top});auto gu=p.gather(w(prefix+"experts.gate_up_proj.weight"),selected,0),down=p.gather(w(prefix+"experts.down_proj.weight"),selected,0);auto input=p.view(hidden,{n,h,1});auto projected=p.matmul(gu,input);auto activated=p.silu_mul(p.view(projected,{n,2*m}));auto output=p.view(p.matmul(down,p.view(activated,{n,m,1})),{n,h});auto scale=p.narrow(route,1,choice,1);contributions.push_back(p.binary(output,scale,'*'));}
    auto out=contributions[0];for(int j=1;j<top;++j)out=p.binary(out,contributions[j],'+');return out;
  }
  std::pair<T,T> forward(Program& p,const T& ids,const T& positions,const T& slots,const T& page_indices,const T& mask,const std::vector<int>& counts,const std::vector<int>& aux_layers){
    int rows=ids->shape[0];auto hidden=p.gather(embed,ids,0);T residual;std::vector<T> aux;std::vector<int64_t> heads(hq);for(int j=0;j<hq;++j)heads[j]=j/(hq/hkv);auto head_indices=p.indices(heads);
    for(int l=0;l<layers;++l){auto prefix="layers."+std::to_string(l)+".";if(std::find(aux_layers.begin(),aux_layers.end(),l)!=aux_layers.end())aux.push_back(residual?p.binary(hidden,residual,'+'):hidden);
      auto sum=residual?p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+'):p.cast(hidden,ACL_FLOAT);residual=p.cast(sum,ACL_FLOAT16);hidden=p.norm(sum,w(prefix+"input_layernorm.weight"),eps);
      auto q=p.norm(p.view(p.linear(hidden,w(prefix+"self_attn.q_proj.weight")),{rows,hq,dim}),w(prefix+"self_attn.q_norm.weight"),eps),k=p.norm(p.view(p.linear(hidden,w(prefix+"self_attn.k_proj.weight")),{rows,hkv,dim}),w(prefix+"self_attn.k_norm.weight"),eps);q=rotate(p,q,positions,hq);k=rotate(p,k,positions,hkv);auto v=p.linear(hidden,w(prefix+"self_attn.v_proj.weight"));p.scatter(keys[l],slots,p.view(k,{rows,hkv*dim}));p.scatter(values[l],slots,v);
      std::vector<T> attention;int offset=0;for(size_t b=0;b<counts.size();++b){int context=page_indices->shape[1],count=counts[b];auto indices=p.view(p.narrow(page_indices,0,b,1),{context});auto key=p.gather(keys[l],indices,0),value=p.gather(values[l],indices,0);key=p.gather(p.view(key,{context,hkv,dim}),head_indices,1);value=p.gather(p.view(value,{context,hkv,dim}),head_indices,1);auto qs=p.transpose(p.narrow(q,0,offset,count),0,1),ks=p.transpose(p.transpose(key,0,1),1,2),vs=p.transpose(value,0,1);auto logits=p.matmul(qs,ks);logits=p.binary(logits,p.constant(1.f/std::sqrt(float(dim)),ACL_FLOAT16),'*');auto bias=p.view(p.narrow(mask,0,offset,count),{1,count,context});auto probs=p.cast(p.softmax(p.binary(logits,bias,'+')),ACL_FLOAT16);attention.push_back(p.transpose(p.matmul(probs,vs),0,1));offset+=count;}
      auto output=p.cat(attention,0);hidden=p.linear(p.view(output,{rows,hq*dim}),w(prefix+"self_attn.o_proj.weight"));sum=p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+');residual=p.cast(sum,ACL_FLOAT16);hidden=p.norm(sum,w(prefix+"post_attention_layernorm.weight"),eps);hidden=mlp(p,hidden,l);
    }
    return {p.norm(p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+'),w("norm.weight"),eps),aux.empty()?T{}:p.cat(aux,1)};
  }
};
struct Entry {
  Program program;std::vector<int> counts;T ids,positions,slots,pages,mask,tokens,aux;int rows=0;
  Entry(Net& net,Net* context_draft,const Config& c,std::vector<int> sizes,bool draft,bool prefill,const std::vector<int>& aux_layers):counts(std::move(sizes)){
    rows=std::accumulate(counts.begin(),counts.end(),0);ids=program.out({rows},ACL_INT64);positions=program.out({rows},ACL_INT64);slots=program.out({rows},ACL_INT64);pages=program.out({int64_t(counts.size()),c.max_model_len},ACL_INT64);mask=program.out({rows,c.max_model_len});
    auto [hidden,captured]=net.forward(program,ids,positions,slots,pages,mask,counts,draft?std::vector<int>{}:aux_layers);aux=captured;
    if(prefill&&context_draft)context_draft->project(program,aux,positions,slots);
    if(draft){std::vector<T> predictions;int offset=0;for(int count:counts){predictions.push_back(program.narrow(hidden,0,offset+1,count-1));offset+=count;}hidden=program.cat(predictions,0);}
    tokens=program.argmax(program.linear(hidden,net.head));program.finish();
  }
  void prepare(const std::vector<Seq>& sequences,const Config& c,bool prefill,bool draft,const std::vector<int64_t>& proposals={},int64_t mask_token=0){
    std::vector<int64_t> token_ids,position_ids,physical_slots,indices(counts.size()*c.max_model_len);std::vector<uint16_t> biases(rows*c.max_model_len);int offset=0;
    for(size_t b=0;b<counts.size();++b){bool real=b<sequences.size();int count=counts[b];int64_t start=real?(prefill?sequences[b]->cached:sequences[b]->tokens.size()-1):0,length=start+count;require(length<=c.max_model_len,"Ascend 查询越过上下文");
      for(int j=0;j<count;++j){int64_t pos=start+j,id=real?(prefill?sequences[b]->tokens[pos]:sequences[b]->tokens.back()):0;if(draft&&j)id=mask_token;else if(!prefill&&!proposals.empty()&&j)id=proposals.at(b*(count-1)+j-1);token_ids.push_back(id);position_ids.push_back(pos);physical_slots.push_back(real?int64_t(sequences[b]->blocks.at(pos/c.kvcache_block_size))*c.kvcache_block_size+pos%c.kvcache_block_size:c.num_kvcache_blocks*c.kvcache_block_size+j);for(int t=0;t<c.max_model_len;++t)biases[(offset+j)*c.max_model_len+t]=fp16(t<(draft?length:pos+1)?0.f:-65504.f);}
      for(int t=0;t<c.max_model_len;++t)indices[b*c.max_model_len+t]=real&&t<length?int64_t(sequences[b]->blocks.at(t/c.kvcache_block_size))*c.kvcache_block_size+t%c.kvcache_block_size:c.num_kvcache_blocks*c.kvcache_block_size;
      offset+=count;
    }
    upload(ids,token_ids.data(),token_ids.size()*8);upload(positions,position_ids.data(),position_ids.size()*8);upload(slots,physical_slots.data(),physical_slots.size()*8);upload(pages,indices.data(),indices.size()*8);upload(mask,biases.data(),biases.size()*2);
  }
};
struct Projection {Program program;T aux,positions,slots;Projection(Net& net,int count,int width){aux=program.out({count,width});positions=program.out({count},ACL_INT64);slots=program.out({count},ACL_INT64);net.project(program,aux,positions,slots);program.finish();}};
class AscendModel final:public DeviceModel {
  Runtime runtime_;Config config_;std::unique_ptr<Net> target_,draft_;std::vector<int> aux_layers_;int k_=0;uint64_t decode_=0,draft_replays_=0,verify_=0,eager_=0;
  std::map<int,std::unique_ptr<Entry>> decode_graphs_,draft_graphs_,verify_graphs_;std::map<int,std::unique_ptr<Projection>> projections_;std::unique_ptr<Entry> prefill_,eager_draft_,eager_verify_;
  void capture(std::map<int,std::unique_ptr<Entry>>& entries,Net& net,int count,bool draft){for(int b:graph_buckets(config_.max_num_seqs)){auto entry=std::make_unique<Entry>(net,nullptr,config_,std::vector<int>(b,count),draft,false,aux_layers_);entry->prepare({},config_,false,draft,{},draft?draft_->config.at("mask_token_id").get<int64_t>():0);entry->program.capture(runtime_.stream);entries.emplace(b,std::move(entry));}}
  Entry& graph(std::map<int,std::unique_ptr<Entry>>& entries,int n){auto it=entries.lower_bound(n);require(it!=entries.end(),"ACL Graph 桶缺失");return *it->second;}
  ModelOutput standard(const Batch& batch){std::vector<int> counts;for(auto& s:batch.sequences)counts.push_back(batch.prefill?s->scheduled:1);Entry* entry;
    if(!batch.prefill&&!config_.enforce_eager){entry=&graph(decode_graphs_,counts.size());++decode_;}else{if(!prefill_||prefill_->counts!=counts)prefill_=std::make_unique<Entry>(*target_,draft_.get(),config_,counts,false,true,aux_layers_);entry=prefill_.get();++eager_;}
    entry->prepare(batch.sequences,config_,batch.prefill,false);entry->program.run(runtime_.stream);auto ids=read_ids(entry->tokens,runtime_.stream);ModelOutput result;int offset=0;for(size_t b=0;b<batch.sequences.size();++b){auto s=batch.sequences[b];bool complete=!batch.prefill||s->cached+s->scheduled==s->tokens.size();result.tokens.push_back(complete?std::vector<int64_t>{ids[offset+counts[b]-1]}:std::vector<int64_t>{});offset+=counts[b];}return result;
  }
  ModelOutput speculative(const std::vector<Seq>& sequences){int n=sequences.size(),rows=k_+1;Entry *d,*v;
    if(config_.enforce_eager){std::vector<int> counts(n,rows);if(!eager_draft_||eager_draft_->counts!=counts){eager_draft_=std::make_unique<Entry>(*draft_,nullptr,config_,counts,true,false,std::vector<int>{});eager_verify_=std::make_unique<Entry>(*target_,nullptr,config_,counts,false,false,aux_layers_);}d=eager_draft_.get();v=eager_verify_.get();++eager_;}
    else{d=&graph(draft_graphs_,n);v=&graph(verify_graphs_,n);++draft_replays_;++verify_;}
    d->prepare(sequences,config_,false,true,{},draft_->config.at("mask_token_id"));d->program.run(runtime_.stream);auto draft_ids=read_ids(d->tokens,runtime_.stream);v->prepare(sequences,config_,false,false,draft_ids);v->program.run(runtime_.stream);auto verify_ids=read_ids(v->tokens,runtime_.stream);ModelOutput out;
    int width=target_->h*aux_layers_.size();for(int b=0;b<n;++b){std::vector<int64_t> proposed(draft_ids.begin()+b*k_,draft_ids.begin()+(b+1)*k_),verified(verify_ids.begin()+b*rows,verify_ids.begin()+(b+1)*rows);auto committed=accept_greedy(proposed,verified);int count=committed.size();auto& p=*projections_.at(count);auto source=static_cast<char*>(v->aux->data())+b*rows*width*2;check(aclrtMemcpy(p.aux->data(),p.aux->size(),source,count*width*2,ACL_MEMCPY_DEVICE_TO_DEVICE),"接受 aux 拷贝");std::vector<int64_t> positions(count),slots(count);for(int j=0;j<count;++j){int64_t pos=sequences[b]->tokens.size()-1+j;positions[j]=pos;slots[j]=int64_t(sequences[b]->blocks.at(pos/config_.kvcache_block_size))*config_.kvcache_block_size+pos%config_.kvcache_block_size;}upload(p.positions,positions.data(),count*8);upload(p.slots,slots.data(),count*8);p.program.run(runtime_.stream);
      out.tokens.push_back(committed);out.rounds.push_back({{"request_id",sequences[b]->id},{"draft_ids",proposed},{"verify_ids",verified},{"accepted_count",count-1},{"committed_ids",committed},{"context_length",sequences[b]->tokens.size()-1}});
    }check(aclrtSynchronizeStream(runtime_.stream),"草稿上下文提交");return out;
  }
public:
  AscendModel(const Bundle& bundle,Config config):config_(std::move(config)){
    require(config_.dtype=="float16","310P 参考后端仅支持 float16");require(!config_.trace_logits,"310P 参考后端尚未实现 logits 导出");require(config_.max_model_len<=bundle.config["target"]["max_position_embeddings"].get<int64_t>(),"模型位置上限不足");
    target_=std::make_unique<Net>(bundle.target_weights,bundle.config.at("target"),config_);
    if(!bundle.config.at("draft").is_null()){auto dc=bundle.config.at("draft");k_=dc.at("num_tokens");require(k_+1<=config_.kvcache_block_size,"DFlash 查询超出 dummy 页");draft_=std::make_unique<Net>(bundle.draft_weights,dc,config_,target_->embed,target_->head);for(int layer:dc.at("target_layer_ids").get<std::vector<int>>())aux_layers_.push_back(layer+1);for(int c=1;c<=k_+1;++c)projections_[c]=std::make_unique<Projection>(*draft_,c,target_->h*aux_layers_.size());}
    if(!config_.enforce_eager){capture(decode_graphs_,*target_,1,false);if(draft_){capture(draft_graphs_,*draft_,k_+1,true);capture(verify_graphs_,*target_,k_+1,false);}}
  }
  ~AscendModel()override{aclrtSetCurrentContext(runtime_.context);aclrtSynchronizeStream(runtime_.stream);}
  ModelOutput run(const Batch& batch)override{runtime_.enter();if(batch.prefill||!draft_)return standard(batch);ModelOutput out;out.tokens.resize(batch.sequences.size());std::vector<Seq> active;std::vector<size_t> indices;for(size_t j=0;j<batch.sequences.size();++j){auto s=batch.sequences[j];if(config_.max_model_len-int64_t(s->tokens.size())>=k_+1){active.push_back(s);indices.push_back(j);}else{auto value=standard(Batch{false,{s}});out.tokens[j]=value.tokens[0];}}if(!active.empty()){auto value=speculative(active);for(size_t j=0;j<indices.size();++j)out.tokens[indices[j]]=value.tokens[j];out.rounds=value.rounds;}return out;}
  void drop(uint64_t)override{/* 分页上下文随 scheduler 的物理块生命周期回收，重填会覆盖有效位置。 */}
  Json snapshot()const override{return {{"backend","ascend310p"},{"attention","aclnn-nd-reference"},{"maturity","source-prepared-unvalidated"},{"decode_graph_replays",decode_},{"draft_graph_replays",draft_replays_},{"verify_graph_replays",verify_},{"eager_rounds",eager_},{"graph_buckets",config_.enforce_eager?std::vector<int64_t>{}:graph_buckets(config_.max_num_seqs)}};}
};
}
std::unique_ptr<DeviceModel> create_ascend_model(const Bundle& bundle,const Config& config){return std::make_unique<AscendModel>(bundle,config);}
}
