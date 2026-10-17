#include "fe/native_cuda.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace fe::cuda {
namespace {
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<typename T> __global__ void norm_kernel(const T* x,const T* residual,const T* weight,T* output,T* new_residual,int width,float eps){
  const int row=blockIdx.x;float sum=0;extern __shared__ float work[];
  for(int j=threadIdx.x;j<width;j+=blockDim.x){float value=float(x[row*width+j]);if(residual)value+=float(residual[row*width+j]);sum+=value*value;}
  work[threadIdx.x]=sum;__syncthreads();for(int stride=blockDim.x/2;stride;stride/=2){if(threadIdx.x<stride)work[threadIdx.x]+=work[threadIdx.x+stride];__syncthreads();}
  const float scale=rsqrtf(work[0]/width+eps);
  for(int j=threadIdx.x;j<width;j+=blockDim.x){float value=float(x[row*width+j]);if(residual)value+=float(residual[row*width+j]);if(new_residual)new_residual[row*width+j]=T(value);output[row*width+j]=T(value*scale*float(weight[j]));}
}
template<typename T> __global__ void silu_kernel(const T* input,T* output,int width,int total){int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=total)return;int row=index/width,col=index%width;float x=float(input[row*width*2+col]),up=float(input[row*width*2+width+col]);output[index]=T((x/(1.f+expf(-x)))*up);}
template<typename T> __global__ void rope_kernel(T* q,T* k,const int64_t* pos,const float* cache,int rows,int hq,int hkv,int dim){
  int index=blockIdx.x*blockDim.x+threadIdx.x;int total=rows*(hq+hkv)*(dim/2);if(index>=total)return;
  int half=dim/2,j=index%half,head=(index/half)%(hq+hkv),row=index/(half*(hq+hkv));bool isq=head<hq;int h=isq?head:head-hq;
  T* base=(isq?q:k)+(row*(isq?hq:hkv)+h)*dim;float a=float(base[j]),b=float(base[j+half]);const float* cs=cache+pos[row]*dim;float c=cs[j],s=cs[j+half];base[j]=T(__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s)));base[j+half]=T(__fadd_rn(__fmul_rn(b,c),__fmul_rn(a,s)));
}
template<typename T> __global__ void store_kernel(const T* k,const T* v,T* kc,T* vc,const int64_t* slots,int rows,int width){int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=rows*width)return;int row=index/width,col=index%width;int64_t slot=slots[row];if(slot<0)return;kc[slot*width+col]=k[index];vc[slot*width+col]=v[index];}
__global__ void accept_kernel(const int64_t* draft,const int64_t* verify,int64_t* out,int k){int b=blockIdx.x;if(threadIdx.x)return;int m=0;while(m<k&&draft[b*k+m]==verify[b*(k+1)+m]){out[b*(k+2)+m]=draft[b*k+m];++m;}out[b*(k+2)+m]=verify[b*(k+1)+m];out[b*(k+2)+k+1]=m+1;}
}
void norm(const void* x,const void* residual,const void* weight,void* output,void* new_residual,int rows,int width,float eps,bool bf16,void* stream){
  auto s=static_cast<cudaStream_t>(stream);
  if(bf16)norm_kernel<<<rows,256,256*sizeof(float),s>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(residual),static_cast<const __nv_bfloat16*>(weight),static_cast<__nv_bfloat16*>(output),static_cast<__nv_bfloat16*>(new_residual),width,eps);
  else norm_kernel<<<rows,256,256*sizeof(float),s>>>(static_cast<const half*>(x),static_cast<const half*>(residual),static_cast<const half*>(weight),static_cast<half*>(output),static_cast<half*>(new_residual),width,eps);check(cudaGetLastError());
}
void silu_mul(const void* gu,void* output,int rows,int width,bool bf16,void* stream){int total=rows*width;auto s=static_cast<cudaStream_t>(stream);if(bf16)silu_kernel<<<(total+255)/256,256,0,s>>>(static_cast<const __nv_bfloat16*>(gu),static_cast<__nv_bfloat16*>(output),width,total);else silu_kernel<<<(total+255)/256,256,0,s>>>(static_cast<const half*>(gu),static_cast<half*>(output),width,total);check(cudaGetLastError());}
void rope(void* q,void* k,const int64_t* pos,const float* cache,int rows,int hq,int hkv,int dim,bool bf16,void* stream,int){int blocks=(rows*(hq+hkv)*(dim/2)+255)/256;auto s=static_cast<cudaStream_t>(stream);if(bf16)rope_kernel<<<blocks,256,0,s>>>(static_cast<__nv_bfloat16*>(q),static_cast<__nv_bfloat16*>(k),pos,cache,rows,hq,hkv,dim);else rope_kernel<<<blocks,256,0,s>>>(static_cast<half*>(q),static_cast<half*>(k),pos,cache,rows,hq,hkv,dim);check(cudaGetLastError());}
void store(const void* k,const void* v,void* kc,void* vc,const int64_t* slots,int rows,int width,bool bf16,void* stream,int){auto s=static_cast<cudaStream_t>(stream);int blocks=(rows*width+255)/256;if(bf16)store_kernel<<<blocks,256,0,s>>>(static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),static_cast<__nv_bfloat16*>(kc),static_cast<__nv_bfloat16*>(vc),slots,rows,width);else store_kernel<<<blocks,256,0,s>>>(static_cast<const half*>(k),static_cast<const half*>(v),static_cast<half*>(kc),static_cast<half*>(vc),slots,rows,width);check(cudaGetLastError());}
void accept(const int64_t* draft,const int64_t* verify,int64_t* out,int batch,int k,void* stream){accept_kernel<<<batch,1,0,static_cast<cudaStream_t>(stream)>>>(draft,verify,out,k);check(cudaGetLastError());}
}
