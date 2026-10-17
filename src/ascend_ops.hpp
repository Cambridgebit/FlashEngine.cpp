#pragma once
// 仅由 Ascend 后端包含；公共 engine/scheduler 不包含任何 ACL 类型。
#include "fe/common.hpp"
#include "fe/half.hpp"
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_inplace_copy.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_topk.h>
#include <aclnnop/aclnn_arg_max.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_index_copy.h>
#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>

namespace fe::ascend {
inline void check(int rc,const char* where){if(rc)throw std::runtime_error(std::string(where)+" 失败，状态="+std::to_string(rc));}
inline int64_t elements(const std::vector<int64_t>& shape){int64_t n=1;for(auto d:shape){require(d>0&&n<=INT64_MAX/d,"Ascend shape 溢出");n*=d;}return n;}
inline size_t bytes(aclDataType dtype){return dtype==ACL_FLOAT?4:dtype==ACL_INT64?8:dtype==ACL_INT32?4:2;}
inline uint16_t fp16(float value){return float_to_half(value);}
struct Memory {void* data=nullptr;size_t size=0;explicit Memory(size_t n):size(n){check(aclrtMalloc(&data,n,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc");}~Memory(){if(data)aclrtFree(data);}};
struct Tensor {
  std::shared_ptr<Memory> memory;std::vector<int64_t> shape,stride;int64_t offset=0;aclDataType dtype;aclTensor* handle=nullptr;
  Tensor(std::shared_ptr<Memory> m,std::vector<int64_t> s,aclDataType d,std::vector<int64_t> st={},int64_t off=0):memory(std::move(m)),shape(std::move(s)),stride(std::move(st)),offset(off),dtype(d){
    if(stride.empty()){stride.resize(shape.size());int64_t acc=1;for(int j=int(shape.size())-1;j>=0;--j){stride[j]=acc;acc*=shape[j];}}
    require(stride.size()==shape.size()&&offset>=0,"非法 Ascend view");int64_t last=offset;for(size_t j=0;j<shape.size();++j){require(stride[j]>=0,"不支持负 stride");last+=(shape[j]-1)*stride[j];}require(uint64_t(last)<memory->size/bytes(dtype),"Ascend view 超出存储");
    int64_t storage=memory->size/bytes(dtype);handle=aclCreateTensor(shape.data(),shape.size(),dtype,stride.data(),offset,ACL_FORMAT_ND,&storage,1,memory->data);require(handle,"aclCreateTensor 失败");
  }
  ~Tensor(){if(handle)aclDestroyTensor(handle);}
  void* data()const{return static_cast<char*>(memory->data)+offset*bytes(dtype);}
  size_t size()const{return elements(shape)*bytes(dtype);}
};
using T=std::shared_ptr<Tensor>;
inline T tensor(std::vector<int64_t> shape,aclDataType dtype=ACL_FLOAT16){auto n=elements(shape);return std::make_shared<Tensor>(std::make_shared<Memory>(n*bytes(dtype)),std::move(shape),dtype);}
inline void upload(const T& t,const void* data,size_t size){require(size==t->size(),"Ascend 上传长度不匹配");check(aclrtMemcpy(t->data(),t->size(),data,size,ACL_MEMCPY_HOST_TO_DEVICE),"H2D");}
inline std::vector<int64_t> read_ids(const T& t,aclrtStream stream){check(aclrtSynchronizeStream(stream),"读回同步");std::vector<int64_t> out(elements(t->shape));check(aclrtMemcpy(out.data(),out.size()*8,t->data(),out.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"D2H");return out;}
struct Program {
  using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
  struct Op {Execute execute;uint64_t workspace;aclOpExecutor* executor;};
  std::vector<Op> ops;std::vector<T> retained;std::vector<aclScalar*> scalars;std::vector<aclIntArray*> arrays;
  std::unique_ptr<Memory> workspace;uint64_t workspace_size=0;aclmdlRI graph=nullptr;
  ~Program(){if(graph)aclmdlRIDestroy(graph);for(auto& op:ops)aclDestroyAclOpExecutor(op.executor);for(auto p:arrays)aclDestroyIntArray(p);for(auto p:scalars)aclDestroyScalar(p);}
  template<class F,class... Args>void add(F first,Execute execute,Args... args){uint64_t size=0;aclOpExecutor* executor=nullptr;check(first(args...,&size,&executor),"ACLNN 一阶段");try{check(aclSetAclOpExecutorRepeatable(executor),"executor 复用");ops.push_back({execute,size,executor});workspace_size=std::max(workspace_size,size);}catch(...){aclDestroyAclOpExecutor(executor);throw;}}
  T keep(T value){retained.push_back(value);return value;}
  T out(std::vector<int64_t> shape,aclDataType dtype=ACL_FLOAT16){return keep(tensor(std::move(shape),dtype));}
  T view(const T& x,std::vector<int64_t> shape,std::vector<int64_t> stride={},int64_t offset=0){return keep(std::make_shared<Tensor>(x->memory,std::move(shape),x->dtype,std::move(stride),x->offset+offset));}
  T transpose(const T& x,int a,int b){auto shape=x->shape,stride=x->stride;std::swap(shape[a],shape[b]);std::swap(stride[a],stride[b]);return view(x,shape,stride);}
  T narrow(const T& x,int dim,int64_t begin,int64_t count){auto shape=x->shape;require(begin>=0&&count>0&&begin+count<=shape.at(dim),"Ascend slice 越界");shape[dim]=count;return view(x,shape,x->stride,begin*x->stride[dim]);}
  aclScalar* scalar(float value){auto p=aclCreateScalar(&value,ACL_FLOAT);require(p,"scalar 创建失败");scalars.push_back(p);return p;}
  T cast(const T& x,aclDataType dtype){if(x->dtype==dtype)return x;auto y=out(x->shape,dtype);add(aclnnCastGetWorkspaceSize,aclnnCast,x->handle,dtype,y->handle);return y;}
  T cat(const std::vector<T>& inputs,int dim){auto shape=inputs.at(0)->shape;shape[dim]=0;for(auto& x:inputs)shape[dim]+=x->shape[dim];auto y=out(shape,inputs[0]->dtype);int64_t offset=0;for(auto& x:inputs){auto dst=narrow(y,dim,offset,x->shape[dim]);add(aclnnInplaceCopyGetWorkspaceSize,aclnnInplaceCopy,dst->handle,x->handle);offset+=x->shape[dim];}return y;}
  T matmul(const T& a,const T& b){auto shape=a->shape;shape.back()=b->shape.back();auto y=out(shape,a->dtype);add(aclnnMatmulGetWorkspaceSize,aclnnMatmul,a->handle,b->handle,y->handle,int8_t(0));return y;}
  T linear(const T& x,const T& weight){return matmul(x,transpose(weight,weight->shape.size()-2,weight->shape.size()-1));}
  T binary(const T& a,const T& b,char kind){auto shape=a->shape;if(b->shape.size()>shape.size())shape=b->shape;else for(size_t i=0;i<b->shape.size();++i)shape[shape.size()-1-i]=std::max(shape[shape.size()-1-i],b->shape[b->shape.size()-1-i]);auto y=out(shape,a->dtype);
    if(kind=='+')add(aclnnAddGetWorkspaceSize,aclnnAdd,a->handle,b->handle,scalar(1),y->handle);else if(kind=='-')add(aclnnSubGetWorkspaceSize,aclnnSub,a->handle,b->handle,scalar(1),y->handle);else if(kind=='*')add(aclnnMulGetWorkspaceSize,aclnnMul,a->handle,b->handle,y->handle);else add(aclnnDivGetWorkspaceSize,aclnnDiv,a->handle,b->handle,y->handle);return y;
  }
  T constant(float value,aclDataType dtype){auto x=out({1},dtype);if(dtype==ACL_FLOAT)upload(x,&value,4);else{auto half=fp16(value);upload(x,&half,2);}return x;}
  T indices(const std::vector<int64_t>& values){auto x=out({int64_t(values.size())},ACL_INT64);upload(x,values.data(),values.size()*8);return x;}
  T gather(const T& x,const T& ids,int dim){auto shape=x->shape;shape[dim]=elements(ids->shape);auto y=out(shape,x->dtype);add(aclnnGatherV2GetWorkspaceSize,aclnnGatherV2,x->handle,int64_t(dim),ids->handle,y->handle);return y;}
  void scatter(const T& cache,const T& ids,const T& values){add(aclnnInplaceIndexCopyGetWorkspaceSize,aclnnInplaceIndexCopy,cache->handle,int64_t(0),ids->handle,values->handle);}
  T norm(const T& x,const T& weight,double eps){auto xf=cast(x,ACL_FLOAT),w=cast(weight,ACL_FLOAT),y=out(x->shape,ACL_FLOAT);auto shape=x->shape;shape.back()=1;auto rstd=out(shape,ACL_FLOAT);add(aclnnRmsNormGetWorkspaceSize,aclnnRmsNorm,xf->handle,w->handle,eps,y->handle,rstd->handle);return cast(y,ACL_FLOAT16);}
  T silu_mul(const T& gu){int64_t width=gu->shape.back()/2;auto input=cast(gu,ACL_FLOAT);auto a=narrow(input,input->shape.size()-1,0,width),b=narrow(input,input->shape.size()-1,width,width),activated=out(a->shape,ACL_FLOAT);add(aclnnSiluGetWorkspaceSize,aclnnSilu,a->handle,activated->handle);return cast(binary(activated,b,'*'),ACL_FLOAT16);}
  T softmax(const T& input){auto x=cast(input,ACL_FLOAT),y=out(x->shape,ACL_FLOAT);add(aclnnSoftmaxGetWorkspaceSize,aclnnSoftmax,x->handle,int64_t(-1),y->handle);return y;}
  T sum(const T& x,int dim,bool keepdim){auto axis=int64_t(dim);auto dims=aclCreateIntArray(&axis,1);require(dims,"axis 创建失败");arrays.push_back(dims);auto shape=x->shape;if(keepdim)shape[dim]=1;else shape.erase(shape.begin()+dim);auto y=out(shape,x->dtype);add(aclnnReduceSumGetWorkspaceSize,aclnnReduceSum,x->handle,dims,keepdim,x->dtype,y->handle);return y;}
  std::pair<T,T> topk(const T& x,int64_t k){auto shape=x->shape;shape.back()=k;auto values=out(shape,x->dtype),ids=out(shape,ACL_INT64);add(aclnnTopkGetWorkspaceSize,aclnnTopk,x->handle,k,int64_t(-1),true,true,values->handle,ids->handle);return {values,ids};}
  T argmax(const T& x){auto shape=x->shape;shape.back()=1;auto y=out(shape,ACL_INT64);add(aclnnArgMaxGetWorkspaceSize,aclnnArgMax,x->handle,int64_t(-1),true,y->handle);return y;}
  void finish(){if(workspace_size)workspace=std::make_unique<Memory>(workspace_size);}
  void eager(aclrtStream stream){for(auto& op:ops)check(op.execute(workspace?workspace->data:nullptr,op.workspace,op.executor,stream),"ACLNN 执行");}
  void capture(aclrtStream stream){eager(stream);check(aclrtSynchronizeStream(stream),"Graph 预热");check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"ACL Graph begin");try{eager(stream);check(aclmdlRICaptureEnd(stream,&graph),"ACL Graph end");}catch(...){aclmdlRI abandoned=nullptr;aclmdlRICaptureEnd(stream,&abandoned);if(abandoned)aclmdlRIDestroy(abandoned);throw;}}
  void run(aclrtStream stream){if(graph)check(aclmdlRIExecuteAsync(graph,stream),"ACL Graph replay");else eager(stream);}
};
}
