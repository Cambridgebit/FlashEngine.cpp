// Qwen3.5 文本参考执行：逐 token GDN，独立请求状态槽，固定单行 decode Graph。
#include "fe/model.hpp"
#include "fe/profiles.hpp"
#include "fe/weight_loader.hpp"
#include "fe/block_draft.hpp"
#include "fe/prefix_state.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <map>
#include <set>

namespace fe {
namespace {
using at::Tensor;
struct HybridLayer {
  bool linear=false;std::string prefix;
  Tensor qkv,o,qn,kn,in,post,kc,vc,gu,down,router,w13,w2,shared_gu,shared_down,shared_gate;
  Tensor qkvz,ba,conv,alog,dt,gnorm,conv_state,ssm_state;
};
class HybridModel final:public DeviceModel {
  Config runtime_;Json cfg_;int h_,dim_,hq_,hkv_,nk_,nv_,dk_,dv_,kernel_,rotary_,cd_;
  Tensor embed_,head_,norm_,cos_,sin_,token_,position_,write_slot_,read_slots_,state_slot_,valid_,timeline_,logits_,output_;
  std::vector<HybridLayer> layers_;std::map<uint64_t,int> slots_;std::set<int> available_;
  std::unique_ptr<at::cuda::CUDAGraph> graph_;uint64_t replays_=0,verify_replays_=0,eager_=0;bool bf16_;
  PrefixStateCache prefix_;std::unique_ptr<BlockDraft> draft_;std::vector<int> aux_layers_;Tensor aux_;int k_=0;
  at::TensorOptions options_;bool recurrent_=false;
  Tensor norm(const Tensor& x,const Tensor& w,bool gemma=true){auto v=x.to(at::kFloat);return (v*at::rsqrt((v*v).mean(-1,true)+cfg_.at("rms_norm_eps").get<double>())*(gemma?1+w.to(at::kFloat):w.to(at::kFloat))).to(x.scalar_type());}
  Tensor mlp(const Tensor& x,const Tensor& gu,const Tensor& down){auto both=at::linear(x,gu);auto width=both.size(-1)/2;return at::linear(at::silu(both.slice(-1,0,width))*both.slice(-1,width),down);}
  Tensor feed_forward(const Tensor& x,const HybridLayer& l){
    if(!l.router.defined())return mlp(x,l.gu,l.down);
    auto scores=at::softmax(at::linear(x,l.router).to(at::kFloat),-1);int k=cfg_.at("num_experts_per_tok"),width=cfg_.at("moe_intermediate_size");
    auto [weights,ids]=at::topk(scores,k,-1,true,true);weights=weights/weights.sum(-1,true);
    auto flat=ids.flatten();auto gu=at::bmm(l.w13.index_select(0,flat),x.repeat_interleave(k,0).unsqueeze(-1)).squeeze(-1);auto activation=at::silu(gu.slice(1,0,width))*gu.slice(1,width);
    auto routed=at::bmm(l.w2.index_select(0,flat),activation.unsqueeze(-1)).squeeze(-1).reshape({1,k,h_});auto out=(routed*weights.to(x.scalar_type()).unsqueeze(-1)).sum(1);
    if(l.shared_gu.defined())out=out+mlp(x,l.shared_gu,l.shared_down)*at::sigmoid(at::linear(x,l.shared_gate));return out;
  }
  Tensor full_attention(const Tensor& x,HybridLayer& l){
    auto projected=at::linear(x,l.qkv);auto pair=projected.slice(1,0,2*hq_*dim_).reshape({1,hq_,2*dim_});auto gate=pair.slice(2,dim_);
    auto q=norm(pair.slice(2,0,dim_),l.qn),k=norm(projected.slice(1,2*hq_*dim_,(2*hq_+hkv_)*dim_).reshape({1,hkv_,dim_}),l.kn),v=projected.slice(1,(2*hq_+hkv_)*dim_).reshape({1,hkv_,dim_});
    auto cs=cos_.index_select(0,position_).unsqueeze(1),sn=sin_.index_select(0,position_).unsqueeze(1);
    auto rotate=[&](const Tensor& value){auto a=value.slice(2,0,rotary_/2).to(at::kFloat),b=value.slice(2,rotary_/2,rotary_).to(at::kFloat);return at::cat({(a*cs-b*sn).to(value.scalar_type()),(b*cs+a*sn).to(value.scalar_type()),value.slice(2,rotary_)},2);};q=rotate(q);k=rotate(k);
    l.kc.index_copy_(0,write_slot_,k);l.vc.index_copy_(0,write_slot_,v);
    auto keys=l.kc.index_select(0,read_slots_).repeat_interleave(hq_/hkv_,1).permute({1,2,0});auto values=l.vc.index_select(0,read_slots_).repeat_interleave(hq_/hkv_,1).permute({1,0,2});
    auto score=at::bmm(q.permute({1,0,2}).to(at::kFloat),keys.to(at::kFloat))/std::sqrt(double(dim_));score.masked_fill_(timeline_>=valid_,-INFINITY);
    auto attended=at::bmm(at::softmax(score,-1),values.to(at::kFloat)).permute({1,0,2}).to(x.scalar_type());return at::linear((attended*at::sigmoid(gate)).reshape({1,hq_*dim_}),l.o);
  }
  Tensor linear_attention(const Tensor& x,HybridLayer& l){
    auto projections=at::linear(x,l.qkvz);auto raw=projections.slice(1,0,cd_),z=projections.slice(1,cd_).reshape({1,nv_,dv_});auto ba=at::linear(x,l.ba);auto b=ba.slice(1,0,nv_),a=ba.slice(1,nv_);
    auto previous=l.conv_state.index_select(0,state_slot_);auto window=at::cat({previous,raw.unsqueeze(-1)},-1);
    auto convolved=(window.to(at::kFloat)*l.conv.reshape({1,cd_,kernel_}).to(at::kFloat)).sum(-1).to(x.scalar_type());convolved=at::silu(convolved);
    l.conv_state.index_copy_(0,state_slot_,window.slice(2,1).contiguous());
    auto q=convolved.slice(1,0,nk_*dk_).reshape({1,nk_,dk_}),k=convolved.slice(1,nk_*dk_,2*nk_*dk_).reshape({1,nk_,dk_}),v=convolved.slice(1,2*nk_*dk_).reshape({1,nv_,dv_});
    q=q.repeat_interleave(nv_/nk_,1);k=k.repeat_interleave(nv_/nk_,1);
    auto l2=[](const Tensor& t){return t*at::rsqrt((t*t).sum(-1,true)+1e-6);};q=l2(q).to(at::kFloat)/std::sqrt(double(dk_));k=l2(k).to(at::kFloat);
    auto decay=at::exp(-at::exp(l.alog)*at::softplus(a.to(at::kFloat)+l.dt));auto beta=at::sigmoid(b).to(at::kFloat);
    auto state=l.ssm_state.index_select(0,state_slot_)*decay.unsqueeze(-1).unsqueeze(-1);auto memory=(state*k.unsqueeze(-1)).sum(-2);auto delta=(v.to(at::kFloat)-memory)*beta.unsqueeze(-1);state=state+k.unsqueeze(-1)*delta.unsqueeze(-2);l.ssm_state.index_copy_(0,state_slot_,state);
    auto core=(state*q.unsqueeze(-1)).sum(-2).to(x.scalar_type());auto fp=core.to(at::kFloat);
    auto normalized=(fp*at::rsqrt((fp*fp).mean(-1,true)+cfg_.at("rms_norm_eps").get<double>())).to(core.scalar_type())*l.gnorm;
    auto gate=cfg_.at("output_gate_type")=="sigmoid"?at::sigmoid(z.to(at::kFloat)):at::silu(z.to(at::kFloat));
    return at::linear((normalized*gate).to(x.scalar_type()).reshape({1,nv_*dv_}),l.o);
  }
  void body(){
    auto hidden=at::embedding(embed_,token_);std::map<int,Tensor> captured;int lid=0;
    for(auto& l:layers_){if(std::find(aux_layers_.begin(),aux_layers_.end(),lid)!=aux_layers_.end())captured[lid]=hidden;auto normalized=norm(hidden,l.in);hidden=hidden+(l.linear?linear_attention(normalized,l):full_attention(normalized,l));hidden=hidden+feed_forward(norm(hidden,l.post),l);++lid;}
    if(draft_){if(std::find(aux_layers_.begin(),aux_layers_.end(),lid)!=aux_layers_.end())captured[lid]=hidden;std::vector<Tensor> values;for(int id:aux_layers_)values.push_back(captured.at(id));aux_.copy_(at::cat(values,1));}
    // eager prefill 与 Graph 必须写同一地址，不能在捕获后重新绑定输出 Tensor。
    logits_.copy_(at::linear(norm(hidden,norm_),head_));output_.copy_(at::argmax(logits_,-1));
  }
  Tensor upload(const std::vector<int64_t>& values){return at::from_blob(const_cast<int64_t*>(values.data()),{int64_t(values.size())},at::TensorOptions().dtype(at::kLong)).to(at::kCUDA);}
  using State=std::vector<Tensor>;
  State save_state(int slot){State out;for(auto& l:layers_)if(l.linear){out.push_back(l.conv_state[slot].clone());out.push_back(l.ssm_state[slot].clone());}return out;}
  void restore_state(int slot,const State& saved){size_t i=0;for(auto& l:layers_)if(l.linear){l.conv_state[slot].copy_(saved.at(i++));l.ssm_state[slot].copy_(saved.at(i++));}require(i==saved.size(),"GDN 快照层数不匹配");}
  int state(const Seq& s){if(auto it=slots_.find(s->id);it!=slots_.end())return it->second;require(!available_.empty(),"Qwen3.5 状态槽耗尽");auto slot=*available_.begin();available_.erase(slot);slots_[s->id]=slot;for(auto& l:layers_)if(l.linear){l.conv_state[slot].zero_();l.ssm_state[slot].zero_();}if(s->cached&&recurrent_){auto key=prefix_snapshot_key(s->tokens,s->cached,s->blocks.at((s->cached-1)/runtime_.kvcache_block_size));require(prefix_.has(key),"目标 Prefix 命中但 GDN 快照缺失");restore_state(slot,prefix_.get(key));}return slot;}
  void save_prefix(const Seq& s,const std::vector<int64_t>& path,size_t end,const State& saved){if(recurrent_&&runtime_.enable_prefix_cache&&end%runtime_.kvcache_block_size==0){require(end<=path.size(),"GDN 前缀键越界");prefix_.put(prefix_snapshot_key(path,end,s->blocks.at((end-1)/runtime_.kvcache_block_size)),saved);}}
  void prepare(const Seq& s,size_t pos,int64_t input){
    int slot=state(s);int64_t write=int64_t(s->blocks.at(pos/runtime_.kvcache_block_size))*runtime_.kvcache_block_size+pos%runtime_.kvcache_block_size;
    std::vector<int64_t> indices(runtime_.max_model_len,runtime_.num_kvcache_blocks*runtime_.kvcache_block_size);for(size_t i=0;i<=pos;++i)indices[i]=int64_t(s->blocks.at(i/runtime_.kvcache_block_size))*runtime_.kvcache_block_size+i%runtime_.kvcache_block_size;
    token_.copy_(upload({input}));position_.copy_(upload({int64_t(pos)}));write_slot_.copy_(upload({write}));state_slot_.copy_(upload({slot}));valid_.copy_(upload({int64_t(pos+1)}));read_slots_.copy_(upload(indices));
  }
  void execute(bool graph,bool verify=false){if(graph&&graph_){graph_->replay();if(verify)++verify_replays_;else ++replays_;}else{body();++eager_;}}
  ModelOutput speculate(const Seq& s){
    auto proposed=draft_->propose(s);size_t start=s->tokens.size()-1;std::vector<int64_t> verified;std::vector<State> states;std::vector<Tensor> auxiliary;
    // 每行消费前一行状态，并保留验证后状态；接受之后只恢复 count-1 行。
    for(int j=0;j<=k_;++j){prepare(s,start+j,j?proposed.at(j-1):s->tokens.back());execute(true,true);verified.push_back(output_.item<int64_t>());states.push_back(save_state(slots_.at(s->id)));auxiliary.push_back(aux_.clone());}
    auto committed=accept_greedy(proposed,verified);size_t count=committed.size();restore_state(slots_.at(s->id),states.at(count-1));
    auto path=s->tokens;path.insert(path.end(),proposed.begin(),proposed.begin()+count-1);
    for(size_t j=0;j<count;++j){draft_->project(auxiliary[j],s,start+j);save_prefix(s,path,start+j+1,states[j]);}
    ModelOutput out;out.tokens.push_back(committed);out.rounds.push_back({{"request_id",s->id},{"draft_ids",proposed},{"verify_ids",verified},{"accepted_count",count-1},{"committed_ids",committed},{"context_length",start}});return out;
  }
public:
  HybridModel(const Bundle& bundle,Config runtime):runtime_(runtime),cfg_(bundle.config.at("target")),bf16_(runtime.dtype=="bfloat16"),prefix_(runtime.enable_prefix_cache?runtime.recurrent_prefix_cache_bytes:0){
    c10::InferenceMode inference;at::cuda::set_device(0);for(auto& kind:cfg_.at("layer_types"))if(kind=="linear_attention")recurrent_=true;
    require(runtime.max_model_len<=cfg_.at("max_position_embeddings").get<int64_t>(),"Qwen3.5 上下文越界");
    h_=cfg_.at("hidden_size");dim_=cfg_.at("head_dim");hq_=cfg_.at("num_attention_heads");hkv_=cfg_.at("num_key_value_heads");nk_=cfg_.at("linear_num_key_heads");nv_=cfg_.at("linear_num_value_heads");dk_=cfg_.at("linear_key_head_dim");dv_=cfg_.at("linear_value_head_dim");kernel_=cfg_.at("linear_conv_kernel_dim");rotary_=cfg_.at("rotary_dim");cd_=2*nk_*dk_+nv_*dv_;
    require(cfg_.at("mamba_ssm_dtype")=="float32"&&(cfg_.at("output_gate_type")=="silu"||cfg_.at("output_gate_type")=="sigmoid"),"不支持的 GDN 状态 dtype/gate");
    options_=at::TensorOptions().device(at::kCUDA).dtype(bf16_?at::kBFloat16:at::kHalf);auto load=[&](const std::string& name){return load_gpu_weight(bundle.target_weights,name,options_);};
    embed_=load("embed_tokens.weight");head_=cfg_.at("tie_word_embeddings").get<bool>()?embed_:load("lm_head.weight");norm_=load("norm.weight");
    if(!bundle.config.at("draft").is_null()){auto dc=bundle.config.at("draft");require(dc.value("spec_method",std::string())=="dflash","Qwen3.5 当前只支持 DFlash 草稿");k_=dc.at("num_tokens");for(int id:dc.at("target_layer_ids").get<std::vector<int>>())aux_layers_.push_back(id+1);draft_=create_block_draft(bundle,runtime,embed_,head_);aux_=at::zeros({1,int64_t(aux_layers_.size())*h_},options_);}
    auto f=options_.dtype(at::kFloat);auto inv=1.0/at::pow(cfg_.at("rope_theta").get<double>(),at::arange(0,rotary_,2,f)/rotary_);auto freqs=at::arange(runtime.max_model_len,f).unsqueeze(1)*inv.unsqueeze(0);cos_=freqs.cos();sin_=freqs.sin();
    for(int lid=0;lid<cfg_.at("num_hidden_layers").get<int>();++lid){
      HybridLayer l;auto p="layers."+std::to_string(lid)+".";l.prefix=p;l.in=load(p+"input_layernorm.weight");l.post=load(p+"post_attention_layernorm.weight");l.linear=cfg_.at("layer_types")[lid]=="linear_attention";
      if(l.linear){auto g=p+"linear_attn.";l.qkvz=at::cat({load(g+"in_proj_qkv.weight"),load(g+"in_proj_z.weight")},0);l.ba=at::cat({load(g+"in_proj_b.weight"),load(g+"in_proj_a.weight")},0);l.conv=load(g+"conv1d.weight");l.alog=load_gpu_weight(bundle.target_weights,g+"A_log",f);l.dt=load_gpu_weight(bundle.target_weights,g+"dt_bias",f);l.gnorm=load(g+"norm.weight");l.o=load(g+"out_proj.weight");l.conv_state=at::zeros({runtime.max_num_seqs+1,cd_,kernel_-1},options_);l.ssm_state=at::zeros({runtime.max_num_seqs+1,nv_,dk_,dv_},f);}
      else{auto a=p+"self_attn.";l.qkv=at::cat({load(a+"q_proj.weight"),load(a+"k_proj.weight"),load(a+"v_proj.weight")},0);l.o=load(a+"o_proj.weight");l.qn=load(a+"q_norm.weight");l.kn=load(a+"k_norm.weight");l.kc=at::zeros({(runtime.num_kvcache_blocks+1)*runtime.kvcache_block_size,hkv_,dim_},options_);l.vc=at::zeros_like(l.kc);}
      if(moe_layer(cfg_,lid)){
        l.router=load(p+"mlp.gate.weight");if(bundle.target_weights.contains(p+"mlp.experts.gate_up_proj.weight")){l.w13=load(p+"mlp.experts.gate_up_proj.weight");l.w2=load(p+"mlp.experts.down_proj.weight");}
        else{std::vector<Tensor> gu,down;for(int e=0;e<cfg_.at("num_experts").get<int>();++e){auto ep=p+"mlp.experts."+std::to_string(e)+".";gu.push_back(at::cat({load(ep+"gate_proj.weight"),load(ep+"up_proj.weight")},0));down.push_back(load(ep+"down_proj.weight"));}l.w13=at::stack(gu);l.w2=at::stack(down);}
        l.shared_gu=at::cat({load(p+"mlp.shared_expert.gate_proj.weight"),load(p+"mlp.shared_expert.up_proj.weight")},0);l.shared_down=load(p+"mlp.shared_expert.down_proj.weight");l.shared_gate=load(p+"mlp.shared_expert_gate.weight");
      }else{l.gu=at::cat({load(p+"mlp.gate_proj.weight"),load(p+"mlp.up_proj.weight")},0);l.down=load(p+"mlp.down_proj.weight");}layers_.push_back(std::move(l));
    }
    for(int i=0;i<runtime.max_num_seqs;++i)available_.insert(i);
    auto longs=options_.dtype(at::kLong);token_=at::zeros({1},longs);position_=at::zeros_like(token_);write_slot_=at::full_like(token_,runtime.num_kvcache_blocks*runtime.kvcache_block_size);state_slot_=at::full_like(token_,runtime.max_num_seqs);valid_=at::ones_like(token_);read_slots_=at::full({runtime.max_model_len},runtime.num_kvcache_blocks*runtime.kvcache_block_size,longs);timeline_=at::arange(runtime.max_model_len,longs);
    logits_=at::zeros({1,cfg_.at("vocab_size").get<int64_t>()},options_);output_=at::zeros_like(token_);
    if(!runtime.enforce_eager){auto side=at::cuda::getStreamFromPool();at::cuda::CUDAStreamGuard guard(side);at::cuda::device_synchronize();body();body();at::cuda::device_synchronize();graph_=std::make_unique<at::cuda::CUDAGraph>();graph_->capture_begin();body();graph_->capture_end();graph_->replay();at::cuda::device_synchronize();}
  }
  ModelOutput run(const Batch& batch)override{
    c10::InferenceMode inference;ModelOutput out;
    // 先恢复整批命中状态，再允许新快照驱逐，避免后续请求的状态被提前淘汰。
    for(auto& s:batch.sequences)state(s);
    for(auto& s:batch.sequences){size_t start=batch.prefill?s->cached:s->tokens.size()-1;size_t rows=batch.prefill?s->scheduled:1;
      if(!batch.prefill&&draft_&&s->tokens.size()+k_+1<=size_t(runtime_.max_model_len)){auto value=speculate(s);out.tokens.push_back(std::move(value.tokens[0]));for(auto& round:value.rounds)out.rounds.push_back(std::move(round));continue;}
      for(size_t pos=start;pos<start+rows;++pos){prepare(s,pos,s->tokens.at(pos));execute(!batch.prefill);if(draft_)draft_->project(aux_,s,pos);if(runtime_.enable_prefix_cache&&(pos+1)%runtime_.kvcache_block_size==0)save_prefix(s,s->tokens,pos+1,save_state(slots_.at(s->id)));}
      bool complete=!batch.prefill||start+rows==s->tokens.size();out.tokens.push_back(complete?std::vector<int64_t>{output_.item<int64_t>()}:std::vector<int64_t>{});
      if(complete&&runtime_.trace_logits){auto [vals,ids]=at::topk(logits_.flatten().to(at::kFloat),8);auto v=vals.to(at::kCPU).contiguous();auto i=ids.to(at::kCPU).contiguous();out.samples.push_back({{"token_index",s->tokens.size()-s->prompt_size},{"ids",std::vector<int64_t>(i.data_ptr<int64_t>(),i.data_ptr<int64_t>()+8)},{"values",std::vector<float>(v.data_ptr<float>(),v.data_ptr<float>()+8)}});}
    }return out;
  }
  void drop(uint64_t id)override{if(auto it=slots_.find(id);it!=slots_.end()){available_.insert(it->second);slots_.erase(it);}}
  bool prefix_page_ready(const std::vector<int64_t>& key,int32_t page)const override{return !recurrent_||prefix_.has(prefix_snapshot_key(key,key.size(),page));}
  void evict_prefix_page(const std::vector<int64_t>& key,int32_t page)override{prefix_.erase(prefix_snapshot_key(key,key.size(),page));}
  Json snapshot()const override{Json out{{"backend",runtime_.backend},{"attention","qwen35-aten-recurrent-reference"},{"target_profile",cfg_["model_type"]},{"spec_method",draft_?"dflash":"none"},{"decode_graph_replays",replays_},{"draft_graph_replays",0},{"verify_graph_replays",verify_replays_},{"eager_rounds",eager_},{"graph_buckets",graph_?std::vector<int>{1}:std::vector<int>{}},{"graph_execution","sequential-single-row"},{"recurrent_active_requests",slots_.size()},{"prefix_state",prefix_.snapshot()},{"quantization",cfg_.value("quantization",std::string("none"))},{"quantized_compute",false}};if(draft_)out.update(draft_->snapshot());return out;}
};
}
std::unique_ptr<DeviceModel> create_hybrid_model(const Bundle& bundle,const Config& config){return std::make_unique<HybridModel>(bundle,config);}
}
