#include "fe/native_cuda.hpp"
#include "fe/flashinfer_compat.cuh"
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <stdexcept>
#include <memory>
#include <vector>

namespace fe::cuda {
using namespace flashinfer;
namespace {void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}}
struct AttentionPlan {
  int hq,hkv,dim,page,max_batch,max_pages,batch=0;bool bf16;
  void* floats=nullptr;void* ints=nullptr;void* pinned=nullptr;
  int32_t *qo=nullptr,*kv=nullptr,*pages=nullptr,*last=nullptr;
  PrefillPlanInfo info;
  std::vector<uint64_t> graph_layout;
  ~AttentionPlan(){cudaFree(floats);cudaFree(ints);cudaFreeHost(pinned);cudaFree(qo);cudaFree(kv);cudaFree(pages);cudaFree(last);}
};
AttentionPlan* create_plan(int hq,int hkv,int dim,int page,int max_batch,int max_pages,bool bf16,int){
  if(dim!=128)throw std::runtime_error("原生 CUDA v1 attention 仅编译 head_dim=128");
  auto plan=std::make_unique<AttentionPlan>();auto& p=*plan;p.hq=hq;p.hkv=hkv;p.dim=dim;p.page=page;p.max_batch=max_batch;p.max_pages=max_pages;p.bf16=bf16;
  check(cudaMalloc(&p.floats,32<<20));check(cudaMalloc(&p.ints,8<<20));check(cudaMallocHost(&p.pinned,8<<20));
  check(cudaMalloc(&p.qo,(max_batch+1)*sizeof(int32_t)));check(cudaMalloc(&p.kv,(max_batch+1)*sizeof(int32_t)));check(cudaMalloc(&p.pages,max_batch*max_pages*sizeof(int32_t)));check(cudaMalloc(&p.last,max_batch*sizeof(int32_t)));return plan.release();
}
void destroy_plan(AttentionPlan* plan){delete plan;}
void prepare_plan(AttentionPlan* ptr,const int32_t* qo,const int32_t* kv,const int32_t* pages,const int32_t* last,int batch,int rows,bool graph,void* stream){
  auto& p=*ptr;auto s=static_cast<cudaStream_t>(stream);if(batch>p.max_batch||kv[batch]>p.max_batch*p.max_pages)throw std::runtime_error("attention 元数据超过容量");
  check(PrefillPlan<int32_t>(p.floats,32<<20,p.ints,p.pinned,8<<20,p.info,const_cast<int32_t*>(qo),const_cast<int32_t*>(kv),rows,batch,p.hq,p.hkv,p.dim,p.dim,p.page,graph,2,-1,-1,true,0,s));
  if(graph){auto& i=p.info;std::vector<uint64_t> layout{uint64_t(i.cta_tile_q),uint64_t(i.padded_batch_size),uint64_t(i.request_indices_offset),uint64_t(i.qo_tile_indices_offset),uint64_t(i.kv_tile_indices_offset),uint64_t(i.o_indptr_offset),uint64_t(i.kv_chunk_size_ptr_offset),uint64_t(i.total_num_rows_offset)};if(!p.graph_layout.empty()&&p.graph_layout!=layout)throw std::runtime_error("FlashInfer 图外 plan 改变了捕获布局，拒绝错误 replay");p.graph_layout=std::move(layout);if(i.split_kv)throw std::runtime_error("当前固定工作区图路径不支持 split KV");}
  check(cudaMemcpyAsync(p.qo,qo,(batch+1)*sizeof(int32_t),cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(p.kv,kv,(batch+1)*sizeof(int32_t),cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(p.pages,pages,kv[batch]*sizeof(int32_t),cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(p.last,last,batch*sizeof(int32_t),cudaMemcpyHostToDevice,s));p.batch=batch;
}
template<typename T,bool Causal,int Tile>void run(AttentionPlan& p,void* q,void* k,void* v,void* output,cudaStream_t stream){
  auto& info=p.info;using Params=BatchPrefillPagedParams<T,T,T,int32_t>;using Variant=DefaultAttention<false,false,false,false>;
  paged_kv_t<T,int32_t> kv(p.hkv,p.page,p.dim,p.batch,QKVLayout::kNHD,static_cast<T*>(k),static_cast<T*>(v),p.pages,p.kv,p.last);
  Params params(static_cast<T*>(q),kv,nullptr,p.qo,nullptr,nullptr,static_cast<T*>(output),nullptr,nullptr,p.hq,p.hq*p.dim,p.dim,-1,0,1.f/sqrtf(float(p.dim)),1,1000000);
  auto base=static_cast<char*>(p.ints);
  params.request_indices=reinterpret_cast<int32_t*>(base+info.request_indices_offset);params.qo_tile_indices=reinterpret_cast<int32_t*>(base+info.qo_tile_indices_offset);params.kv_tile_indices=reinterpret_cast<int32_t*>(base+info.kv_tile_indices_offset);params.o_indptr=reinterpret_cast<int32_t*>(base+info.o_indptr_offset);params.kv_chunk_size_ptr=reinterpret_cast<int32_t*>(base+info.kv_chunk_size_ptr_offset);params.block_valid_mask=info.enable_cuda_graph&&info.split_kv?reinterpret_cast<bool*>(base+info.block_valid_mask_offset):nullptr;params.padded_batch_size=info.padded_batch_size;params.max_total_num_rows=info.total_num_rows;params.total_num_rows=info.enable_cuda_graph?reinterpret_cast<uint32_t*>(base+info.total_num_rows_offset):nullptr;params.partition_kv=false;
  check((BatchPrefillWithPagedKVCacheDispatched<Tile,128,128,PosEncodingMode::kNone,false,Causal?MaskMode::kCausal:MaskMode::kNone,Variant,Params>(params,nullptr,nullptr,false,stream)));
}
template<typename T,bool Causal>void dispatch(AttentionPlan& p,void* q,void* k,void* v,void* o,cudaStream_t stream){switch(p.info.cta_tile_q){case 16:run<T,Causal,16>(p,q,k,v,o,stream);break;case 64:run<T,Causal,64>(p,q,k,v,o,stream);break;case 128:run<T,Causal,128>(p,q,k,v,o,stream);break;default:throw std::runtime_error("未编译的 FlashInfer tile");}}
void attention(AttentionPlan* ptr,void* q,void* k,void* v,void* o,bool causal,void* stream){auto& p=*ptr;auto s=static_cast<cudaStream_t>(stream);if(p.bf16){if(causal)dispatch<__nv_bfloat16,true>(p,q,k,v,o,s);else dispatch<__nv_bfloat16,false>(p,q,k,v,o,s);}else{if(causal)dispatch<half,true>(p,q,k,v,o,s);else dispatch<half,false>(p,q,k,v,o,s);}}
}
