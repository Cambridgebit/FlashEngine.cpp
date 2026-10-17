// CUDA 兼容 LibTorch 的全 C++ 参考算子。CoreX 不编译 NVIDIA PTX/FlashInfer。
#include "fe/native_cuda.hpp"
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fe::cuda {
namespace {
using at::Tensor;
auto opts(bool bf16){return at::TensorOptions().device(at::kCUDA).dtype(bf16?at::kBFloat16:at::kHalf);}
Tensor view(const void* p,at::IntArrayRef shape,bool bf16){return at::from_blob(const_cast<void*>(p),shape,opts(bf16));}
Tensor longs(const void* p,at::IntArrayRef shape){return at::from_blob(const_cast<void*>(p),shape,at::TensorOptions().device(at::kCUDA).dtype(at::kLong));}
Tensor upload(const std::vector<int64_t>& v){return at::from_blob(const_cast<int64_t*>(v.data()),{int64_t(v.size())},at::TensorOptions().dtype(at::kLong)).to(at::kCUDA);}
}
struct AttentionPlan {
  int hq,hkv,dim,page,max_batch,max_pages,cache_slots,max_context,batch=0,rows=0;bool bf16;
  std::vector<int32_t> qo;
  Tensor indices,causal_mask,full_mask;
};
AttentionPlan* create_plan(int hq,int hkv,int dim,int page,int max_batch,int max_pages,bool bf16,int blocks){
  auto p=std::make_unique<AttentionPlan>();p->hq=hq;p->hkv=hkv;p->dim=dim;p->page=page;p->max_batch=max_batch;p->max_pages=max_pages;p->bf16=bf16;p->cache_slots=(blocks+1)*page;p->max_context=(max_pages-1)*page;return p.release();
}
void destroy_plan(AttentionPlan* p){delete p;}
void prepare_plan(AttentionPlan* p,const int32_t* qo,const int32_t* kv,const int32_t* pages,const int32_t* last,int batch,int rows,bool graph,void*){
  if(batch>p->max_batch)throw std::runtime_error("参考 attention 批次越界");
  std::vector<int64_t> indices(batch*p->max_context);std::vector<uint8_t> causal(rows*p->max_context),full(causal.size());
  for(int b=0;b<batch;++b){int length=(kv[b+1]-kv[b]-1)*p->page+last[b],qrows=qo[b+1]-qo[b];if(length<qrows||length>p->max_context)throw std::runtime_error("参考 attention 上下文越界");
    for(int t=0;t<p->max_context;++t){indices[b*p->max_context+t]=t<length?int64_t(pages[kv[b]+t/p->page])*p->page+t%p->page:p->cache_slots-1;for(int row=qo[b];row<qo[b+1];++row){full[row*p->max_context+t]=t<length;causal[row*p->max_context+t]=t<=length-qrows+row-qo[b];}}
  }
  auto upload_mask=[&](std::vector<uint8_t>& data){return at::from_blob(data.data(),{rows,p->max_context},at::TensorOptions().dtype(at::kByte)).to(at::kCUDA).to(at::kBool);};
  auto ids=upload(indices).reshape({batch,p->max_context});auto cm=upload_mask(causal),fm=upload_mask(full);
  if(graph&&p->indices.defined()){if(p->batch!=batch||p->rows!=rows)throw std::runtime_error("参考图形状发生变化");p->indices.copy_(ids);p->causal_mask.copy_(cm);p->full_mask.copy_(fm);}
  else{p->indices=ids;p->causal_mask=cm;p->full_mask=fm;}
  p->batch=batch;p->rows=rows;p->qo.assign(qo,qo+batch+1);
}
void attention(AttentionPlan* p,void* q,void* k,void* v,void* output,bool causal,void*){
  auto query=view(q,{p->rows,p->hq,p->dim},p->bf16),keys=view(k,{p->cache_slots,p->hkv,p->dim},p->bf16),values=view(v,{p->cache_slots,p->hkv,p->dim},p->bf16),out=view(output,{p->rows,p->hq,p->dim},p->bf16);
  for(int b=0;b<p->batch;++b){auto key=keys.index_select(0,p->indices[b]).repeat_interleave(p->hq/p->hkv,1).permute({1,2,0});auto value=values.index_select(0,p->indices[b]).repeat_interleave(p->hq/p->hkv,1).permute({1,0,2});auto x=query.slice(0,p->qo[b],p->qo[b+1]).permute({1,0,2}).to(at::kFloat);
    auto logits=at::bmm(x,key.to(at::kFloat))/std::sqrt(double(p->dim));auto mask=(causal?p->causal_mask:p->full_mask).slice(0,p->qo[b],p->qo[b+1]).unsqueeze(0);logits.masked_fill_(mask.logical_not(),-INFINITY);auto result=at::bmm(at::softmax(logits,-1),value.to(at::kFloat));out.slice(0,p->qo[b],p->qo[b+1]).copy_(result.permute({1,0,2}));
  }
}
void norm(const void* x,const void* residual,const void* weight,void* output,void* new_residual,int rows,int width,float eps,bool bf16,void*){
  auto value=view(x,{rows,width},bf16).to(at::kFloat);if(residual)value=value+view(residual,{rows,width},bf16).to(at::kFloat);if(new_residual)view(new_residual,{rows,width},bf16).copy_(value);
  auto normalized=value*at::rsqrt((value*value).mean(-1,true)+eps);view(output,{rows,width},bf16).copy_(normalized*view(weight,{width},bf16).to(at::kFloat));
}
void silu_mul(const void* gu,void* output,int rows,int width,bool bf16,void*){auto input=view(gu,{rows,2*width},bf16).to(at::kFloat);view(output,{rows,width},bf16).copy_(at::silu(input.slice(1,0,width))*input.slice(1,width));}
void rope(void* q,void* k,const int64_t* pos,const float* cache,int rows,int hq,int hkv,int dim,bool bf16,void*,int max_positions){
  auto all=at::from_blob(const_cast<float*>(cache),{max_positions,dim},at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).index_select(0,longs(pos,{rows})).unsqueeze(1);auto cs=all.slice(2,0,dim/2),sn=all.slice(2,dim/2);
  auto apply=[&](void* data,int heads){auto dst=view(data,{rows,heads,dim},bf16);auto value=dst.to(at::kFloat),a=value.slice(2,0,dim/2),b=value.slice(2,dim/2);dst.copy_(at::cat({a*cs-b*sn,b*cs+a*sn},2));};apply(q,hq);apply(k,hkv);
}
void store(const void* k,const void* v,void* kc,void* vc,const int64_t* slots,int rows,int width,bool bf16,void*,int cache_slots){auto ids=longs(slots,{rows});ids=at::where(ids<0,cache_slots-1,ids);view(kc,{cache_slots,width},bf16).index_copy_(0,ids,view(k,{rows,width},bf16));view(vc,{cache_slots,width},bf16).index_copy_(0,ids,view(v,{rows,width},bf16));}
void accept(const int64_t* draft,const int64_t* verify,int64_t* committed,int batch,int k,void*){
  auto d=longs(draft,{batch,k}),v=longs(verify,{batch,k+1}),out=longs(committed,{batch,k+2});auto count=at::cumprod((d==v.slice(1,0,k)).to(at::kLong),1).sum(1);auto indices=at::arange(k,v.options()).unsqueeze(0);out.slice(1,0,k).copy_(at::where(indices<count.unsqueeze(1),d,v.slice(1,0,k)));out.select(1,k).copy_(v.select(1,k));out.scatter_(1,count.unsqueeze(1),v.gather(1,count.unsqueeze(1)));out.select(1,k+1).copy_(count+1);
}
}
