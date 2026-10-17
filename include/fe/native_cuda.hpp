#pragma once
#include <cstdint>
#include <cstddef>
// 此头文件不依赖 LibTorch，仅声明原生 CUDA 算子边界。
namespace fe::cuda {
struct AttentionPlan;
AttentionPlan* create_plan(int hq,int hkv,int dim,int page,int max_batch,int max_pages,bool bf16,int blocks);
void destroy_plan(AttentionPlan* plan);
void prepare_plan(AttentionPlan* plan,const int32_t* qo,const int32_t* kv,const int32_t* pages,const int32_t* last,int batch,int rows,bool graph,void* stream);
void attention(AttentionPlan* plan,void* q,void* k,void* v,void* output,bool causal,void* stream);
void norm(const void* x,const void* residual,const void* weight,void* output,void* new_residual,int rows,int width,float eps,bool bf16,void* stream);
void silu_mul(const void* gu,void* output,int rows,int width,bool bf16,void* stream);
void rope(void* q,void* k,const int64_t* pos,const float* cache,int rows,int hq,int hkv,int dim,bool bf16,void* stream,int max_positions);
void store(const void* k,const void* v,void* kc,void* vc,const int64_t* slots,int rows,int width,bool bf16,void* stream,int cache_slots);
void accept(const int64_t* draft,const int64_t* verify,int64_t* committed,int batch,int k,void* stream);
}
