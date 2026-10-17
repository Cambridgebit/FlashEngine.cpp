#pragma once
// 当前 FlashInfer 的 fast_mod_div 依赖高于 CUDA 13.0 所带 CCCL。
// 在本地适配边界提供完全等价的无状态整数运算，不修改全局依赖。
// 性能资格仍由实际回归决定；该实现不假定除数为 2 次幂。
#define FLASHINFER_FASTDIV_CUH_
#include <cuda_runtime.h>
#include <cstdint>
namespace flashinfer {
struct uint_fastdiv {
  uint32_t d_;
  __host__ __device__ uint_fastdiv():d_(0){}
  __host__ uint_fastdiv(uint32_t d):d_(d){}
  __host__ __device__ operator unsigned int()const{return d_;}
  __host__ __device__ void divmod(uint32_t n,uint32_t& q,uint32_t& r)const{uint32_t divisor=d_?d_:1;q=n/divisor;r=n-q*divisor;}
};
__host__ __device__ inline uint32_t operator/(uint32_t n,const uint_fastdiv& d){uint32_t q,r;d.divmod(n,q,r);return q;}
__host__ __device__ inline uint32_t operator%(uint32_t n,const uint_fastdiv& d){uint32_t q,r;d.divmod(n,q,r);return r;}
}
