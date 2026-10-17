// Qwen3.5 的原生 DFlash 参考草稿。固定块 Graph，非因果注意力，任意合法 head_dim。
#include "fe/block_draft.hpp"
#include "fe/weight_loader.hpp"
#include "fe/native_cuda.hpp"
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
namespace fe { namespace {
using at::Tensor;
struct DraftLayer {Tensor qkv,o,gu,down,in,post,qn,kn,kc,vc;};
class ReferenceBlockDraft final:public BlockDraft {
  Config runtime_;Json cfg_;int h_,hq_,hkv_,dim_,rows_,width_;bool bf16_;
  Tensor embed_,head_,norm_,fc_,hidden_norm_,cos_,sin_,ids_,positions_,slots_,read_,valid_,timeline_,proposals_;
  std::vector<DraftLayer> layers_;std::unique_ptr<at::cuda::CUDAGraph> graph_;uint64_t replays_=0,eager_=0,projected_=0;
  Tensor upload(const std::vector<int64_t>& v){return at::from_blob(const_cast<int64_t*>(v.data()),{int64_t(v.size())},at::TensorOptions().dtype(at::kLong)).to(at::kCUDA);}
  Tensor norm(const Tensor& x,const Tensor& weight){auto dense=x.contiguous(),out=at::empty_like(dense);cuda::norm(dense.data_ptr(),nullptr,weight.data_ptr(),out.data_ptr(),nullptr,dense.numel()/dense.size(-1),dense.size(-1),cfg_.at("rms_norm_eps"),bf16_,at::cuda::getCurrentCUDAStream().stream());return out;}
  std::pair<Tensor,Tensor> added(const Tensor& x,const Tensor& residual,const Tensor& weight){auto out=at::empty_like(x),sum=at::empty_like(x);cuda::norm(x.data_ptr(),residual.data_ptr(),weight.data_ptr(),out.data_ptr(),sum.data_ptr(),x.numel()/x.size(-1),x.size(-1),cfg_.at("rms_norm_eps"),bf16_,at::cuda::getCurrentCUDAStream().stream());return {out,sum};}
  Tensor rotate(const Tensor& x,const Tensor& positions){auto a=x.slice(-1,0,dim_/2).to(at::kFloat),b=x.slice(-1,dim_/2).to(at::kFloat);auto cs=cos_.index_select(0,positions).unsqueeze(1),sn=sin_.index_select(0,positions).unsqueeze(1);return at::cat({a*cs-b*sn,b*cs+a*sn},-1).to(x.scalar_type());}
  void body(){
    auto hidden=at::embedding(embed_,ids_);Tensor residual;
    for(auto& l:layers_){
      if(!residual.defined()){residual=hidden;hidden=norm(hidden,l.in);}else std::tie(hidden,residual)=added(hidden,residual,l.in);
      auto qkv=at::linear(hidden,l.qkv);auto q=rotate(norm(qkv.slice(1,0,hq_*dim_).reshape({rows_,hq_,dim_}),l.qn),positions_);
      auto k=rotate(norm(qkv.slice(1,hq_*dim_,(hq_+hkv_)*dim_).reshape({rows_,hkv_,dim_}),l.kn),positions_);auto v=qkv.slice(1,(hq_+hkv_)*dim_).reshape({rows_,hkv_,dim_});
      l.kc.index_copy_(0,slots_,k);l.vc.index_copy_(0,slots_,v);
      auto keys=l.kc.index_select(0,read_).repeat_interleave(hq_/hkv_,1).permute({1,2,0});auto values=l.vc.index_select(0,read_).repeat_interleave(hq_/hkv_,1).permute({1,0,2});
      auto scores=at::bmm(q.permute({1,0,2}).to(at::kFloat),keys.to(at::kFloat))/std::sqrt(double(dim_));scores.masked_fill_(timeline_>=valid_,-INFINITY);
      auto attended=at::bmm(at::softmax(scores,-1),values.to(at::kFloat)).permute({1,0,2}).to(hidden.scalar_type());hidden=at::linear(attended.reshape({rows_,hq_*dim_}),l.o);
      std::tie(hidden,residual)=added(hidden,residual,l.post);auto gu=at::linear(hidden,l.gu);auto activation=at::empty({rows_,width_},hidden.options());cuda::silu_mul(gu.data_ptr(),activation.data_ptr(),rows_,width_,bf16_,at::cuda::getCurrentCUDAStream().stream());hidden=at::linear(activation,l.down);
    }
    std::tie(hidden,residual)=added(hidden,residual,norm_);auto logits=at::linear(hidden.slice(0,1),head_);proposals_.copy_(at::argmax(logits,-1));
  }
public:
  ReferenceBlockDraft(const Bundle& b,Config runtime,Tensor embed,Tensor head):runtime_(runtime),cfg_(b.config.at("draft")),bf16_(runtime.dtype=="bfloat16"){
    c10::InferenceMode inference;require(cfg_.value("spec_method",std::string("dflash"))=="dflash","hybrid 当前只支持 DFlash1");
    h_=cfg_.at("hidden_size");hq_=cfg_.at("num_attention_heads");hkv_=cfg_.at("num_key_value_heads");dim_=cfg_.at("head_dim");width_=cfg_.at("intermediate_size");rows_=cfg_.at("num_tokens").get<int>()+1;
    require(rows_<=runtime.kvcache_block_size,"DFlash 块超过 KV 页大小");auto options=embed.options();auto load=[&](std::string key){return load_gpu_weight(b.draft_weights,key,options);};
    embed_=b.draft_weights.contains("embed_tokens.weight")?load("embed_tokens.weight"):embed;head_=b.draft_weights.contains("lm_head.weight")?load("lm_head.weight"):head;norm_=load("norm.weight");fc_=load("fc.weight");hidden_norm_=load("hidden_norm.weight");
    auto floats=options.dtype(at::kFloat);auto inv=1.0/at::pow(cfg_.at("rope_theta").get<double>(),at::arange(0,dim_,2,floats)/dim_);auto freq=at::arange(runtime.max_model_len,floats).unsqueeze(1)*inv.unsqueeze(0);cos_=freq.cos();sin_=freq.sin();
    for(int lid=0;lid<cfg_.at("num_hidden_layers").get<int>();++lid){auto p="layers."+std::to_string(lid)+".";DraftLayer l;l.qkv=at::cat({load(p+"self_attn.q_proj.weight"),load(p+"self_attn.k_proj.weight"),load(p+"self_attn.v_proj.weight")},0);l.o=load(p+"self_attn.o_proj.weight");l.qn=load(p+"self_attn.q_norm.weight");l.kn=load(p+"self_attn.k_norm.weight");l.in=load(p+"input_layernorm.weight");l.post=load(p+"post_attention_layernorm.weight");l.gu=at::cat({load(p+"mlp.gate_proj.weight"),load(p+"mlp.up_proj.weight")},0);l.down=load(p+"mlp.down_proj.weight");l.kc=at::zeros({(runtime.num_kvcache_blocks+1)*runtime.kvcache_block_size,hkv_,dim_},options);l.vc=at::zeros_like(l.kc);layers_.push_back(std::move(l));}
    auto longs=options.dtype(at::kLong);ids_=at::zeros({rows_},longs);positions_=at::arange(rows_,longs);slots_=positions_+runtime.num_kvcache_blocks*runtime.kvcache_block_size;read_=at::full({runtime.max_model_len},runtime.num_kvcache_blocks*runtime.kvcache_block_size,longs);valid_=at::full({1},rows_,longs);timeline_=at::arange(runtime.max_model_len,longs);proposals_=at::zeros({rows_-1},longs);
    if(!runtime.enforce_eager){auto side=at::cuda::getStreamFromPool();at::cuda::CUDAStreamGuard guard(side);at::cuda::device_synchronize();body();body();at::cuda::device_synchronize();graph_=std::make_unique<at::cuda::CUDAGraph>();graph_->capture_begin();body();graph_->capture_end();graph_->replay();at::cuda::device_synchronize();}
  }
  void project(const Tensor& aux,const Seq& s,size_t pos)override{
    auto x=norm(at::linear(aux,fc_),hidden_norm_);auto position=upload({int64_t(pos)});auto slot=upload({int64_t(s->blocks.at(pos/runtime_.kvcache_block_size))*runtime_.kvcache_block_size+int64_t(pos%runtime_.kvcache_block_size)});
    for(auto& l:layers_){auto kv=at::linear(x,l.qkv.slice(0,hq_*dim_));auto k=rotate(norm(kv.slice(1,0,hkv_*dim_).reshape({1,hkv_,dim_}),l.kn),position);auto v=kv.slice(1,hkv_*dim_).reshape({1,hkv_,dim_});l.kc.index_copy_(0,slot,k);l.vc.index_copy_(0,slot,v);}++projected_;
  }
  std::vector<int64_t> propose(const Seq& s)override{
    auto start=s->tokens.size()-1,end=start+rows_;require(end<=size_t(runtime_.max_model_len),"DFlash 草稿越界");std::vector<int64_t> ids(rows_,cfg_.at("mask_token_id")),positions,slots,read(runtime_.max_model_len,runtime_.num_kvcache_blocks*runtime_.kvcache_block_size);ids[0]=s->tokens.back();
    for(size_t p=0;p<end;++p){auto slot=int64_t(s->blocks.at(p/runtime_.kvcache_block_size))*runtime_.kvcache_block_size+p%runtime_.kvcache_block_size;read[p]=slot;if(p>=start){positions.push_back(p);slots.push_back(slot);}}
    ids_.copy_(upload(ids));positions_.copy_(upload(positions));slots_.copy_(upload(slots));read_.copy_(upload(read));valid_.fill_(end);if(graph_){graph_->replay();++replays_;}else{body();++eager_;}
    auto cpu=proposals_.to(at::kCPU).contiguous();auto ptr=cpu.data_ptr<int64_t>();return {ptr,ptr+rows_-1};
  }
  Json snapshot()const override{return {{"draft_graph_replays",replays_},{"draft_eager_rounds",eager_},{"draft_projected_tokens",projected_},{"draft_attention","aten-fixed-context-reference"}};}
};
}
std::unique_ptr<BlockDraft> create_block_draft(const Bundle& b,const Config& c,Tensor embed,Tensor head){return std::make_unique<ReferenceBlockDraft>(b,c,embed,head);}
}
