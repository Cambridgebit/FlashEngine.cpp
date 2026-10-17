#include "fe/model.hpp"
#include "fe/native_cuda.hpp"
#include "fe/server.hpp"
#include "fe/profiles.hpp"
#include "fe/weight_loader.hpp"
#include "fe/prefix_state.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#ifndef FE_GPU_BACKEND
#define FE_GPU_BACKEND "cuda"
#endif
#ifndef FE_ATTENTION_ID
#define FE_ATTENTION_ID "flashinfer-native"
#endif

namespace fe {
namespace {
using at::Tensor;
void* stream(){return at::cuda::getCurrentCUDAStream().stream();}
Tensor device_i64(const std::vector<int64_t>& values){return at::from_blob(const_cast<int64_t*>(values.data()),{int64_t(values.size())},at::TensorOptions().dtype(at::kLong)).to(at::kCUDA);}
std::vector<int64_t> host_i64(const Tensor& value){auto cpu=value.to(at::kCPU).contiguous();auto* data=cpu.data_ptr<int64_t>();return {data,data+cpu.numel()};}
Json top_logits(const Tensor& logits,int offset,int rows,const std::string& stage){auto [values,indices]=at::topk(logits.slice(0,offset,offset+rows).to(at::kFloat),8);auto cpu=values.to(at::kCPU).contiguous();auto ids=host_i64(indices);Json idrows=Json::array(),valrows=Json::array();for(int r=0;r<rows;++r){idrows.push_back(std::vector<int64_t>(ids.begin()+r*8,ids.begin()+(r+1)*8));auto p=cpu.data_ptr<float>()+r*8;valrows.push_back(std::vector<float>(p,p+8));}return {{"stage",stage},{"ids",idrows},{"values",valrows}};}
struct Layer {Tensor qkv,o,gu,down,in,post,qn,kn,kc,vc,router,w13,w2;};
struct Net {
  Json cfg;bool bf16,eagle;int h,hq,hkv,dim,layers,intermediate;
  Tensor embed,head,norm,rope,fc,hidden_norm,fused_kv,d2t;std::vector<Layer> blocks;
  Net(const std::map<std::string,WeightView>& source,Json config,const Config& runtime,Tensor shared_embed={},Tensor shared_head={}):cfg(std::move(config)),bf16(runtime.dtype=="bfloat16"){
    h=cfg.at("hidden_size");hq=cfg.at("num_attention_heads");hkv=cfg.at("num_key_value_heads");dim=cfg.at("head_dim");layers=cfg.at("num_hidden_layers");intermediate=cfg.at("intermediate_size");eagle=cfg.value("spec_method",std::string())=="eagle3";
    require(dim==128,"CUDA 首版仅支持 head_dim=128");
    auto options=at::TensorOptions().device(at::kCUDA).dtype(bf16?at::kBFloat16:at::kHalf);
    auto load=[&](const std::string& name)->Tensor {
      require(source.contains(name),"缺失模型权重："+name);return load_gpu_weight(source,name,options);
    };
    embed=source.contains("embed_tokens.weight")?load("embed_tokens.weight"):shared_embed;
    require(embed.defined(),"缺少 embedding");
    head=source.contains("lm_head.weight")?load("lm_head.weight"):shared_head.defined()?shared_head:embed;
    norm=load("norm.weight");
    auto floats=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto inv=1.0/at::pow(cfg.at("rope_theta").get<double>(),at::arange(0,dim,2,floats)/dim);
    auto freqs=at::arange(runtime.max_model_len,floats).unsqueeze(1)*inv.unsqueeze(0);
    rope=at::cat({freqs.cos(),freqs.sin()},1).contiguous();
    std::vector<Tensor> kvweights;
    for(int lid=0;lid<layers;++lid){
      auto prefix=eagle?std::string("midlayer."):"layers."+std::to_string(lid)+".";Layer layer;
      auto q=load(prefix+"self_attn.q_proj.weight"),k=load(prefix+"self_attn.k_proj.weight"),v=load(prefix+"self_attn.v_proj.weight");
      layer.qkv=at::cat({q,k,v},0);kvweights.push_back(at::cat({k,v},0));
      layer.o=load(prefix+"self_attn.o_proj.weight");if(!eagle){layer.qn=load(prefix+"self_attn.q_norm.weight");layer.kn=load(prefix+"self_attn.k_norm.weight");}
      if(moe_layer(cfg,lid)){
        layer.router=load(prefix+"mlp.gate.weight");
        if(source.contains(prefix+"mlp.experts.gate_up_proj.weight")){layer.w13=load(prefix+"mlp.experts.gate_up_proj.weight");layer.w2=load(prefix+"mlp.experts.down_proj.weight");}
        else{std::vector<Tensor> gates,downs;for(int e=0;e<cfg.at("num_experts").get<int>();++e){auto p=prefix+"mlp.experts."+std::to_string(e)+".";gates.push_back(at::cat({load(p+"gate_proj.weight"),load(p+"up_proj.weight")},0));downs.push_back(load(p+"down_proj.weight"));}layer.w13=at::stack(gates);layer.w2=at::stack(downs);}
      }else{layer.gu=at::cat({load(prefix+"mlp.gate_proj.weight"),load(prefix+"mlp.up_proj.weight")},0);layer.down=load(prefix+"mlp.down_proj.weight");}
      layer.in=load(prefix+"input_layernorm.weight");layer.post=load(prefix+"post_attention_layernorm.weight");
      layer.kc=at::zeros({runtime.num_kvcache_blocks+1,runtime.kvcache_block_size,hkv,dim},options);layer.vc=at::zeros_like(layer.kc);blocks.push_back(std::move(layer));
    }
    if(eagle){fc=load("fc.weight");hidden_norm=load("midlayer.hidden_norm.weight");d2t=load_gpu_weight(source,"d2t",at::TensorOptions().device(at::kCUDA).dtype(at::kLong));}
    else if(source.contains("fc.weight")){fc=load("fc.weight");hidden_norm=load("hidden_norm.weight");fused_kv=at::cat(kvweights,0);}
  }
  Tensor normalized(const Tensor& input,const Tensor& weight){auto dense=input.contiguous();auto out=at::empty_like(dense);cuda::norm(dense.data_ptr(),nullptr,weight.data_ptr(),out.data_ptr(),nullptr,dense.numel()/dense.size(-1),dense.size(-1),cfg.at("rms_norm_eps"),bf16,stream());return out;}
  std::pair<Tensor,Tensor> added(const Tensor& input,const Tensor& residual,const Tensor& weight){auto out=at::empty_like(input),sum=at::empty_like(input);cuda::norm(input.data_ptr(),residual.data_ptr(),weight.data_ptr(),out.data_ptr(),sum.data_ptr(),input.numel()/input.size(-1),input.size(-1),cfg.at("rms_norm_eps"),bf16,stream());return {out,sum};}
  Tensor feed_forward(const Tensor& hidden,const Layer& layer){
    if(!layer.router.defined()){auto gu=at::linear(hidden,layer.gu);auto activated=at::empty({gu.size(0),intermediate},gu.options());cuda::silu_mul(gu.data_ptr(),activated.data_ptr(),gu.size(0),intermediate,bf16,stream());return at::linear(activated,layer.down);}
    // 固定形状 gather/bmm 路径：路由无 host 读回，Graph 不依赖专家动态 token 数。
    auto scores=at::softmax(at::linear(hidden,layer.router).to(at::kFloat),-1);int topk=cfg.at("num_experts_per_tok"),width=cfg.at("moe_intermediate_size");
    auto [weights,ids]=at::topk(scores,topk,-1,true,true);if(cfg.at("norm_topk_prob").get<bool>())weights=weights/weights.sum(-1,true);
    auto flat_ids=ids.flatten();auto input=hidden.repeat_interleave(topk,0).unsqueeze(-1);auto output=at::zeros_like(hidden);auto token_ids=at::floor_divide(at::arange(flat_ids.numel(),ids.options()),topk);auto route_weights=weights.to(hidden.scalar_type()).flatten();
    int64_t bytes_per_row=int64_t(h)*width*6;int64_t chunk=std::max<int64_t>(1,(256LL<<20)/bytes_per_row);
    for(int64_t start=0;start<flat_ids.numel();start+=chunk){int64_t end=std::min(start+chunk,flat_ids.numel());auto selected=flat_ids.slice(0,start,end);auto gu=at::bmm(layer.w13.index_select(0,selected),input.slice(0,start,end)).squeeze(-1);auto activated=at::silu(gu.slice(1,0,width))*gu.slice(1,width);auto down=at::bmm(layer.w2.index_select(0,selected),activated.unsqueeze(-1)).squeeze(-1);output.index_add_(0,token_ids.slice(0,start,end),down*route_weights.slice(0,start,end).unsqueeze(-1));}
    return output;
  }
  Tensor eagle_forward(const Tensor& ids,const Tensor& positions,const Tensor& slots,cuda::AttentionPlan* plan,const Tensor& seed){
    auto& layer=blocks[0];auto input=at::cat({normalized(at::embedding(embed,ids),layer.in),normalized(seed,hidden_norm)},-1);
    auto qkv=at::linear(input,layer.qkv);auto q=qkv.slice(1,0,hq*dim).reshape({-1,hq,dim}).contiguous();auto key=qkv.slice(1,hq*dim,(hq+hkv)*dim).reshape({-1,hkv,dim}).contiguous();auto value=qkv.slice(1,(hq+hkv)*dim).reshape({-1,hkv,dim}).contiguous();
    cuda::rope(q.data_ptr(),key.data_ptr(),positions.data_ptr<int64_t>(),rope.data_ptr<float>(),ids.numel(),hq,hkv,dim,bf16,stream(),rope.size(0));
    cuda::store(key.data_ptr(),value.data_ptr(),layer.kc.data_ptr(),layer.vc.data_ptr(),slots.data_ptr<int64_t>(),ids.numel(),hkv*dim,bf16,stream(),layer.kc.size(0)*layer.kc.size(1));
    auto output=at::empty_like(q);cuda::attention(plan,q.data_ptr(),layer.kc.data_ptr(),layer.vc.data_ptr(),output.data_ptr(),true,stream());
    auto residual=seed+at::linear(output.reshape({-1,hq*dim}),layer.o);
    return residual+feed_forward(normalized(residual,layer.post),layer);
  }
  std::pair<Tensor,Tensor> forward(const Tensor& ids,const Tensor& positions,const Tensor& slots,cuda::AttentionPlan* plan,bool draft,const std::vector<int>& aux_layers){
    auto hidden=at::embedding(embed,ids);Tensor residual;std::map<int,Tensor> captured;
    for(int lid=0;lid<layers;++lid){auto& layer=blocks[lid];
      if(std::find(aux_layers.begin(),aux_layers.end(),lid)!=aux_layers.end())captured[lid]=residual.defined()?hidden+residual:hidden;
      if(!residual.defined()){residual=hidden;hidden=normalized(hidden,layer.in);}else std::tie(hidden,residual)=added(hidden,residual,layer.in);
      auto qkv=at::linear(hidden,layer.qkv);auto q=normalized(qkv.slice(1,0,hq*dim).reshape({-1,hq,dim}),layer.qn);
      auto k=normalized(qkv.slice(1,hq*dim,(hq+hkv)*dim).reshape({-1,hkv,dim}),layer.kn);
      auto v=qkv.slice(1,(hq+hkv)*dim).reshape({-1,hkv,dim}).contiguous();
      cuda::rope(q.data_ptr(),k.data_ptr(),positions.data_ptr<int64_t>(),rope.data_ptr<float>(),ids.numel(),hq,hkv,dim,bf16,stream(),rope.size(0));
      cuda::store(k.data_ptr(),v.data_ptr(),layer.kc.data_ptr(),layer.vc.data_ptr(),slots.data_ptr<int64_t>(),ids.numel(),hkv*dim,bf16,stream(),layer.kc.size(0)*layer.kc.size(1));
      auto output=at::empty_like(q);cuda::attention(plan,q.data_ptr(),layer.kc.data_ptr(),layer.vc.data_ptr(),output.data_ptr(),!draft,stream());
      hidden=at::linear(output.reshape({-1,hq*dim}),layer.o);std::tie(hidden,residual)=added(hidden,residual,layer.post);
      hidden=feed_forward(hidden,layer);
    }
    std::tie(hidden,residual)=added(hidden,residual,norm);std::vector<Tensor> aux;for(int lid:aux_layers)aux.push_back(captured.at(lid));return {hidden,aux.empty()?Tensor():at::cat(aux,1)};
  }
  void project(const Tensor& aux,const Tensor& positions,const Tensor& slots){
    auto x=normalized(at::linear(aux,fc),hidden_norm);auto kv=at::linear(x,fused_kv).reshape({-1,layers,2,hkv,dim});
    for(int lid=0;lid<layers;++lid){auto k=normalized(kv.select(1,lid).select(1,0),blocks[lid].kn);auto v=kv.select(1,lid).select(1,1).contiguous();
      // RoPE 两个参数不可别名；单独的空张量用于被丢弃的第二输出。
      auto unused=k.clone();cuda::rope(k.data_ptr(),unused.data_ptr(),positions.data_ptr<int64_t>(),rope.data_ptr<float>(),aux.size(0),hkv,hkv,dim,bf16,stream(),rope.size(0));
      cuda::store(k.data_ptr(),v.data_ptr(),blocks[lid].kc.data_ptr(),blocks[lid].vc.data_ptr(),slots.data_ptr<int64_t>(),aux.size(0),hkv*dim,bf16,stream(),blocks[lid].kc.size(0)*blocks[lid].kc.size(1));
    }
  }
};
struct Entry {
  int batch,rows;int64_t dummy_page;Tensor ids,positions,slots,tokens,aux,logits,seed,hidden;
  std::unique_ptr<cuda::AttentionPlan,decltype(&cuda::destroy_plan)> plan{nullptr,cuda::destroy_plan};
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  Entry(Net& net,const Config& c,int b,int r):batch(b),rows(r),dummy_page(c.num_kvcache_blocks){
    auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);ids=at::zeros({b*r},options);positions=at::zeros_like(ids);slots=at::full_like(ids,-1);
    if(net.eagle)seed=at::zeros({b*r,net.h},net.embed.options());
    plan.reset(cuda::create_plan(net.hq,net.hkv,net.dim,c.kvcache_block_size,c.max_num_seqs,(c.max_model_len+c.kvcache_block_size-1)/c.kvcache_block_size+1,net.bf16,c.num_kvcache_blocks));
  }
};
class CudaModel final:public DeviceModel {
  Config config_,draft_config_;std::unique_ptr<Net> target_,draft_;std::vector<int> aux_layers_;
  PrefixStateCache prefix_;std::map<uint64_t,int> eagle_slots_;std::set<int> eagle_free_;
  struct AuxPage {Tensor values;size_t begin,filled;};std::map<uint64_t,AuxPage> eagle_pending_;
  int64_t eagle_pages_=0;uint64_t eagle_prefix_tokens_=0;
  std::map<int,std::unique_ptr<Entry>> decode_graphs_,verify_graphs_,draft_graphs_,refill_graphs_;
  std::map<uint64_t,Tensor> eagle_aux_,eagle_carry_;std::map<uint64_t,int64_t> eagle_token_;
  std::unique_ptr<Entry> eager_target_,eager_draft_,eager_verify_,eager_cold_;
  uint64_t decode_replay_=0,draft_replay_=0,verify_replay_=0,eager_=0;
  int k_=0;bool tracing_=true,eagle_=false;
  void prepare(Entry& entry,const std::vector<Seq>& seqs,bool prefill,bool graph,bool draft_input=false){
    std::vector<int64_t> ids,positions,slots;std::vector<int32_t> qo{0},kv{0},pages,last;
    const int count=graph?entry.batch:seqs.size();
    for(int i=0;i<count;++i){
      if(i<int(seqs.size())){auto& s=*seqs[i];int rows=prefill?s.scheduled:entry.rows;int64_t start=prefill?s.cached:s.tokens.size()-1;int64_t length=start+rows;require(length<=config_.max_model_len,"verify 越过上下文边界");
        for(int j=0;j<rows;++j){int64_t pos=start+j;ids.push_back(draft_input&&j>0?draft_->cfg.at("mask_token_id").get<int64_t>():prefill?s.tokens[pos]:s.tokens.back());positions.push_back(pos);require(size_t(pos/config_.kvcache_block_size)<s.blocks.size(),"页表未覆盖写槽");slots.push_back(int64_t(s.blocks[pos/config_.kvcache_block_size])*config_.kvcache_block_size+pos%config_.kvcache_block_size);}
        qo.push_back(qo.back()+rows);int n=(length+config_.kvcache_block_size-1)/config_.kvcache_block_size;require(size_t(n)<=s.blocks.size(),"页表未覆盖上下文");pages.insert(pages.end(),s.blocks.begin(),s.blocks.begin()+n);kv.push_back(pages.size());last.push_back((length-1)%config_.kvcache_block_size+1);
      }else{for(int j=0;j<entry.rows;++j){ids.push_back(0);positions.push_back(j);slots.push_back(-1);}qo.push_back(qo.back()+entry.rows);pages.push_back(entry.dummy_page);kv.push_back(pages.size());last.push_back(entry.rows);}
    }
    if(!graph){entry.ids=device_i64(ids);entry.positions=device_i64(positions);entry.slots=device_i64(slots);}else{entry.ids.copy_(device_i64(ids));entry.positions.copy_(device_i64(positions));entry.slots.copy_(device_i64(slots));}
    cuda::prepare_plan(entry.plan.get(),qo.data(),kv.data(),pages.data(),last.data(),count,ids.size(),graph,stream());
  }
  void body(Entry& entry,Net& net,bool draft){
    if(net.eagle){entry.hidden=net.eagle_forward(entry.ids,entry.positions,entry.slots,entry.plan.get(),entry.seed);entry.logits=at::linear(net.normalized(entry.hidden,net.norm),net.head);auto ids=at::argmax(entry.logits,-1);entry.tokens=ids+net.d2t.index_select(0,ids);return;}
    auto [hidden,aux]=net.forward(entry.ids,entry.positions,entry.slots,entry.plan.get(),draft,draft?std::vector<int>{}:aux_layers_);
    if(draft){auto h=hidden.reshape({entry.batch,entry.rows,net.h}).slice(1,1).reshape({-1,net.h});entry.logits=at::linear(h,net.head);}else entry.logits=at::linear(hidden,net.head);
    entry.tokens=at::argmax(entry.logits,-1);
    entry.aux=aux;
  }
  void capture(std::map<int,std::unique_ptr<Entry>>& entries,Net& net,int rows,bool draft){
    auto side=at::cuda::getStreamFromPool();at::cuda::CUDAStreamGuard guard(side);at::cuda::device_synchronize();
    for(auto b:graph_buckets(config_.max_num_seqs)){
      auto entry=std::make_unique<Entry>(net,net.eagle?draft_config_:config_,b,rows);prepare(*entry,{},false,true,draft);
      body(*entry,net,draft);body(*entry,net,draft);at::cuda::device_synchronize();
      entry->graph=std::make_unique<at::cuda::CUDAGraph>();entry->graph->capture_begin();body(*entry,net,draft);entry->graph->capture_end();entry->graph->replay();at::cuda::device_synchronize();entries.emplace(b,std::move(entry));
    }
  }
  Entry& graph_entry(std::map<int,std::unique_ptr<Entry>>& entries,int n){auto it=entries.lower_bound(n);require(it!=entries.end(),"Graph 桶缺失");return *it->second;}
  Seq eagle_view(const Seq& s){
    if(!eagle_slots_.contains(s->id)){require(!eagle_free_.empty(),"EAGLE3 独立 KV 槽耗尽");auto slot=*eagle_free_.begin();eagle_free_.erase(slot);eagle_slots_[s->id]=slot;}
    auto copy=std::make_shared<Sequence>(*s);copy->blocks.clear();int slot=eagle_slots_.at(s->id);for(int64_t p=0;p<eagle_pages_;++p)copy->blocks.push_back(slot*eagle_pages_+p);return copy;
  }
  std::vector<Seq> eagle_views(const std::vector<Seq>& seqs){std::vector<Seq> out;for(auto& s:seqs)out.push_back(eagle_view(s));return out;}
  void eagle_restore(const Seq& s){
    if(eagle_slots_.contains(s->id))return;
    auto view=eagle_view(s);auto& d=*eager_draft_;d.batch=1;d.rows=1;size_t block=config_.kvcache_block_size;
    // shifted draft 的边界行依赖下一个 token，不能直接共享物理页。
    for(size_t start=0;start<s->cached;start+=block){auto key=prefix_snapshot_key(s->tokens,start+block,s->blocks.at((start+block-1)/block));require(prefix_.has(key),"EAGLE3 前缀 aux 快照缺失");auto aux=prefix_.get(key).at(0).to(target_->embed.options());view->cached=start;view->scheduled=block;prepare(d,{view},true,false);d.ids=device_i64({s->tokens.begin()+start+1,s->tokens.begin()+start+block+1});d.seed=at::linear(aux,draft_->fc);body(d,*draft_,false);eagle_prefix_tokens_+=block;}
  }
  void eagle_record(const Seq& s,const std::vector<int64_t>& path,size_t start,const Tensor& auxiliary){
    if(!config_.enable_prefix_cache)return;size_t block=config_.kvcache_block_size;
    auto cpu=auxiliary.to(at::kCPU);
    for(size_t j=0;j<size_t(cpu.size(0));++j){size_t pos=start+j,offset=pos%block;
      if(!offset)eagle_pending_.insert_or_assign(s->id,AuxPage{at::empty({int64_t(block),cpu.size(1)},cpu.options()),pos,0});
      auto it=eagle_pending_.find(s->id);require(it!=eagle_pending_.end()&&it->second.begin+it->second.filled==pos,"EAGLE3 aux 页不连续");auto& page=it->second;page.values[offset].copy_(cpu[j]);++page.filled;
      if(page.filled==block){require(pos+1<=path.size(),"EAGLE3 前缀 token 键越界");prefix_.put(prefix_snapshot_key(path,pos+1,s->blocks.at(pos/block)),{page.values});eagle_pending_.erase(it);}
    }
  }
  ModelOutput standard(const Batch& batch){
    if(eagle_&&batch.prefill)for(auto& s:batch.sequences)eagle_restore(s);
    Entry* entry;
    if(!batch.prefill&&!config_.enforce_eager){entry=&graph_entry(decode_graphs_,batch.sequences.size());prepare(*entry,batch.sequences,false,true);entry->graph->replay();++decode_replay_;}
    else{entry=eager_target_.get();entry->batch=batch.sequences.size();prepare(*entry,batch.sequences,batch.prefill,false);body(*entry,*target_,false);++eager_;}
    auto tokens=host_i64(entry->tokens);ModelOutput out;size_t offset=0;
    for(auto& s:batch.sequences){int rows=batch.prefill?s->scheduled:1;bool complete=!batch.prefill||s->cached+s->scheduled==s->tokens.size();out.tokens.push_back(complete?std::vector<int64_t>{tokens.at(offset+rows-1)}:std::vector<int64_t>{});
      if(complete&&config_.trace_logits&&!draft_){auto [values,indices]=at::topk(entry->logits.select(0,offset+rows-1).to(at::kFloat),8);auto cpu=values.to(at::kCPU).contiguous();std::vector<float> logits(cpu.data_ptr<float>(),cpu.data_ptr<float>()+cpu.numel());out.samples.push_back({{"token_index",s->tokens.size()-s->prompt_size},{"ids",host_i64(indices)},{"values",logits}});}
      if(draft_&&!eagle_){draft_->project(entry->aux.slice(0,offset,offset+rows),entry->positions.slice(0,offset,offset+rows),entry->slots.slice(0,offset,offset+rows));}
      if(eagle_)eagle_record(s,s->tokens,batch.prefill?s->cached:s->tokens.size()-1,entry->aux.slice(0,offset,offset+rows));
      offset+=rows;
    }
    if(eagle_){
      // shifted draft 流：位置 p 消费 tok[p+1] 与 target aux[p]。
      auto& d=*eager_draft_;d.batch=batch.sequences.size();d.rows=1;prepare(d,eagle_views(batch.sequences),batch.prefill,false);
      std::vector<int64_t> shifted;size_t cursor=0;for(auto& s:batch.sequences){int rows=batch.prefill?s->scheduled:1;auto start=batch.prefill?s->cached:s->tokens.size()-1;for(int j=0;j<rows;++j){size_t pos=start+j+1;shifted.push_back(pos<s->tokens.size()?s->tokens[pos]:tokens[cursor+rows-1]);}if(!batch.prefill||start+rows==s->tokens.size()){eagle_aux_[s->id]=entry->aux.slice(0,cursor+rows-1,cursor+rows).clone();eagle_carry_.erase(s->id);eagle_token_.erase(s->id);}cursor+=rows;}
      d.ids=device_i64(shifted);d.seed=at::linear(entry->aux,draft_->fc);body(d,*draft_,false);
    }return out;
  }
  void eagle_prepare(Entry& e,const std::vector<Seq>& seqs,int shift,bool graph){
    std::vector<Seq> views;for(auto& s:seqs){auto copy=std::make_shared<Sequence>(*s);int64_t length=int64_t(s->tokens.size())+shift;require(length>0,"EAGLE3 位置为负");copy->tokens.resize(length);views.push_back(copy);}prepare(e,views,false,graph);
  }
  ModelOutput eagle_round(const std::vector<Seq>& seqs){
    int n=seqs.size(),rows=k_+1;bool graph=!config_.enforce_eager;
    auto draft_seqs=eagle_views(seqs);
    Entry* d=graph?&graph_entry(draft_graphs_,n):eager_draft_.get();Entry* v=graph?&graph_entry(verify_graphs_,n):eager_verify_.get();
    if(!graph){d->batch=v->batch=n;d->rows=1;d->seed=at::zeros({n,draft_->h},draft_->embed.options());}
    auto run_draft=[&](Entry& e,bool refill){if(graph){e.graph->replay();++draft_replay_;}else{body(e,*draft_,false);++eager_;}(void)refill;};
    std::vector<Tensor> seeds;std::vector<int64_t> first(n);std::vector<bool> cold(n);for(int i=0;i<n;++i){auto id=seqs[i]->id;cold[i]=!eagle_carry_.contains(id);require(eagle_aux_.contains(id),"EAGLE3 缺少目标 aux 状态");seeds.push_back(cold[i]?at::linear(eagle_aux_.at(id),draft_->fc):eagle_carry_.at(id));if(!cold[i])first[i]=eagle_token_.at(id);}
    auto seed=at::cat(seeds);auto proposed=at::zeros({d->batch,k_},d->ids.options());
    // 冷请求单独重算最后一行；热请求保持 refill 提供的 carry，不按 batch 下标保存状态。
    std::vector<Seq> cold_seqs;std::vector<Tensor> cold_seed;std::vector<int> cold_index;std::vector<int64_t> last;
    for(int i=0;i<n;++i)if(cold[i]){cold_seqs.push_back(draft_seqs[i]);cold_seed.push_back(seeds[i]);cold_index.push_back(i);last.push_back(seqs[i]->tokens.back());}
    if(!cold_seqs.empty()){auto& c=*eager_cold_;c.batch=cold_seqs.size();eagle_prepare(c,cold_seqs,-1,false);c.ids=device_i64(last);c.seed=at::cat(cold_seed);body(c,*draft_,false);++eager_;auto values=host_i64(c.tokens);for(size_t j=0;j<cold_index.size();++j)first[cold_index[j]]=values[j];}
    // 热请求不得重写上一轮已提交的 shifted KV 行。
    proposed.select(1,0).slice(0,0,n).copy_(device_i64(first));
    for(int step=1;step<k_;++step){eagle_prepare(*d,draft_seqs,step-1,graph);d->ids.copy_(proposed.select(1,step-1));d->seed.zero_();d->seed.slice(0,0,n).copy_(seed);run_draft(*d,false);seed=d->hidden.slice(0,0,n).clone();proposed.select(1,step).copy_(d->tokens);}
    prepare(*v,seqs,false,graph);v->ids.reshape({v->batch,rows}).slice(1,1).copy_(proposed);if(graph){v->graph->replay();++verify_replay_;}else body(*v,*target_,false);
    auto committed=at::zeros({v->batch,rows+1},v->ids.options());cuda::accept(proposed.data_ptr<int64_t>(),v->tokens.data_ptr<int64_t>(),committed.data_ptr<int64_t>(),n,k_,stream());
    auto values=host_i64(committed),drafts=host_i64(proposed),verified=host_i64(v->tokens);
    Entry* refill=graph?&graph_entry(refill_graphs_,n):eager_draft_.get();if(!graph){refill->batch=n;refill->rows=rows;}
    prepare(*refill,draft_seqs,false,graph);auto view=v->tokens.reshape({v->batch,rows});auto ids=at::cat({view.slice(1,1),view.slice(1,k_,k_+1)},1);for(int i=0;i<n;++i){int m=values[i*(rows+1)+rows]-1;ids[i][m].copy_(view[i][m]);}
    refill->ids.copy_(ids.flatten());auto projected=at::linear(v->aux,draft_->fc);if(graph)refill->seed.copy_(projected);else refill->seed=projected;run_draft(*refill,true);
    ModelOutput out;for(int i=0;i<n;++i){int count=values[i*(rows+1)+rows];require(count>=1&&count<=rows,"EAGLE3 接受长度非法");out.tokens.emplace_back(values.begin()+i*(rows+1),values.begin()+i*(rows+1)+count);int row=i*rows+count-1;auto id=seqs[i]->id;eagle_aux_[id]=v->aux.slice(0,row,row+1).clone();eagle_carry_[id]=refill->hidden.slice(0,row,row+1).clone();eagle_token_[id]=host_i64(refill->tokens.slice(0,row,row+1))[0];auto path=seqs[i]->tokens;path.insert(path.end(),drafts.begin()+i*k_,drafts.begin()+i*k_+count-1);eagle_record(seqs[i],path,seqs[i]->tokens.size()-1,v->aux.slice(0,i*rows,i*rows+count));out.rounds.push_back({{"request_id",id},{"draft_ids",std::vector<int64_t>(drafts.begin()+i*k_,drafts.begin()+(i+1)*k_)},{"verify_ids",std::vector<int64_t>(verified.begin()+i*rows,verified.begin()+(i+1)*rows)},{"accepted_count",count-1},{"committed_ids",out.tokens.back()},{"context_length",seqs[i]->tokens.size()-1}});}
    return out;
  }
  ModelOutput speculative(const std::vector<Seq>& seqs){
    Entry *d,*v;int n=seqs.size(),rows=k_+1;
    if(config_.enforce_eager){d=eager_draft_.get();v=eager_verify_.get();d->batch=v->batch=n;prepare(*d,seqs,false,false,true);body(*d,*draft_,true);++eager_;}
    else{d=&graph_entry(draft_graphs_,n);v=&graph_entry(verify_graphs_,n);prepare(*d,seqs,false,true,true);d->graph->replay();++draft_replay_;}
    prepare(*v,seqs,false,!config_.enforce_eager);
    v->ids.reshape({v->batch,rows}).slice(1,1).copy_(d->tokens.reshape({d->batch,k_}));
    if(config_.enforce_eager)body(*v,*target_,false);else{v->graph->replay();++verify_replay_;}
    auto committed=at::zeros({v->batch,rows+1},v->ids.options());cuda::accept(d->tokens.data_ptr<int64_t>(),v->tokens.data_ptr<int64_t>(),committed.data_ptr<int64_t>(),n,k_,stream());
    auto result=host_i64(committed);auto draft_ids=host_i64(d->tokens),verify_ids=host_i64(v->tokens);ModelOutput out;
    for(int i=0;i<n;++i){int count=result[i*(rows+1)+rows];require(count>=1&&count<=rows,"接受计数越界");out.tokens.emplace_back(result.begin()+i*(rows+1),result.begin()+i*(rows+1)+count);
      draft_->project(v->aux.slice(0,i*rows,i*rows+count),v->positions.slice(0,i*rows,i*rows+count),v->slots.slice(0,i*rows,i*rows+count));
      if(tracing_)out.rounds.push_back({{"request_id",seqs[i]->id},{"draft_ids",std::vector<int64_t>(draft_ids.begin()+i*k_,draft_ids.begin()+(i+1)*k_)},{"verify_ids",std::vector<int64_t>(verify_ids.begin()+i*rows,verify_ids.begin()+(i+1)*rows)},{"accepted_count",count-1},{"committed_ids",out.tokens.back()},{"context_length",seqs[i]->tokens.size()-1}});
      if(config_.trace_logits){out.samples.push_back(top_logits(d->logits,i*k_,k_,"draft"));out.samples.push_back(top_logits(v->logits,i*rows,rows,"verify"));}
    }return out;
  }
public:
  CudaModel(const Bundle& bundle,Config config):config_(std::move(config)),draft_config_(config_),prefix_(config_.enable_prefix_cache?config_.recurrent_prefix_cache_bytes:0){
    c10::InferenceMode inference;at::cuda::set_device(0);
    require(config_.max_model_len<=bundle.config["target"]["max_position_embeddings"].get<int64_t>(),"运行上下文超过模型位置上限");
    target_=std::make_unique<Net>(bundle.target_weights,bundle.config.at("target"),config_);
    if(!bundle.config.at("draft").is_null()){auto dc=bundle.config.at("draft");eagle_=dc.value("spec_method",std::string("dflash"))=="eagle3";k_=dc.at("num_tokens");require(k_>0&&(eagle_||k_<dc.at("block_size").get<int>()),"非法投机长度");require(k_+1<=config_.kvcache_block_size,"草稿行数超过页大小");if(eagle_){eagle_pages_=(config_.max_model_len+config_.kvcache_block_size-1)/config_.kvcache_block_size;draft_config_.num_kvcache_blocks=eagle_pages_*config_.max_num_seqs;draft_config_.validate();for(int slot=0;slot<config_.max_num_seqs;++slot)eagle_free_.insert(slot);}draft_=std::make_unique<Net>(bundle.draft_weights,dc,draft_config_,target_->embed,target_->head);if(eagle_)aux_layers_=dc.at("aux_layers").get<std::vector<int>>();else for(int lid:dc.at("target_layer_ids").get<std::vector<int>>())aux_layers_.push_back(lid+1);}
    eager_target_=std::make_unique<Entry>(*target_,config_,config_.max_num_seqs,1);
    if(draft_){eager_draft_=std::make_unique<Entry>(*draft_,draft_config_,config_.max_num_seqs,k_+1);eager_verify_=std::make_unique<Entry>(*target_,config_,config_.max_num_seqs,k_+1);if(eagle_)eager_cold_=std::make_unique<Entry>(*draft_,draft_config_,config_.max_num_seqs,1);}
    if(!config_.enforce_eager){capture(decode_graphs_,*target_,1,false);if(draft_){capture(draft_graphs_,*draft_,eagle_?1:k_+1,!eagle_);capture(verify_graphs_,*target_,k_+1,false);if(eagle_)capture(refill_graphs_,*draft_,k_+1,false);}}
  }
  ModelOutput run(const Batch& batch)override{
    c10::InferenceMode inference;
    if(batch.prefill||!draft_)return standard(batch);
    ModelOutput out;out.tokens.resize(batch.sequences.size());std::vector<Seq> spec;std::vector<size_t> indices;
    for(size_t i=0;i<batch.sequences.size();++i){auto s=batch.sequences[i];if(config_.max_model_len-int64_t(s->tokens.size())>=k_+1){spec.push_back(s);indices.push_back(i);}else{auto value=standard(Batch{false,{s}});out.tokens[i]=std::move(value.tokens[0]);}}
    if(!spec.empty()){auto value=eagle_?eagle_round(spec):speculative(spec);for(size_t i=0;i<indices.size();++i)out.tokens[indices[i]]=std::move(value.tokens[i]);out.rounds=std::move(value.rounds);out.samples=std::move(value.samples);}return out;
  }
  void drop(uint64_t id)override{eagle_aux_.erase(id);eagle_carry_.erase(id);eagle_token_.erase(id);eagle_pending_.erase(id);if(auto it=eagle_slots_.find(id);it!=eagle_slots_.end()){eagle_free_.insert(it->second);eagle_slots_.erase(it);}}
  bool prefix_page_ready(const std::vector<int64_t>& key,int32_t page)const override{return !eagle_||prefix_.has(prefix_snapshot_key(key,key.size(),page));}
  void evict_prefix_page(const std::vector<int64_t>& key,int32_t page)override{if(eagle_)prefix_.erase(prefix_snapshot_key(key,key.size(),page));}
  Json snapshot()const override{return {{"backend",FE_GPU_BACKEND},{"attention",FE_ATTENTION_ID},{"target_profile",target_->cfg.value("model_type",std::string("qwen3"))},{"spec_method",eagle_?"eagle3":draft_?"dflash":"none"},{"eagle_active_requests",eagle_aux_.size()},{"eagle_kv_active_requests",eagle_slots_.size()},{"eagle_prefix_rebuilt_tokens",eagle_prefix_tokens_},{"prefix_state",prefix_.snapshot()},{"quantization",target_->cfg.value("quantization",std::string("none"))},{"quantized_compute",false},{"decode_graph_replays",decode_replay_},{"draft_graph_replays",draft_replay_},{"verify_graph_replays",verify_replay_},{"eager_rounds",eager_},{"graph_buckets",config_.enforce_eager?std::vector<int64_t>{}:graph_buckets(config_.max_num_seqs)}};}
};
}
std::unique_ptr<DeviceModel> create_cuda_model(const Bundle& bundle,const Config& config){if(bundle.config.at("target").value("model_type",std::string()).starts_with("qwen3_5"))return create_hybrid_model(bundle,config);return std::make_unique<CudaModel>(bundle,config);}
}
