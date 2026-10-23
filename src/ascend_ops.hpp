#pragma once
// 仅由 Ascend 后端包含；公共 engine/scheduler 不包含任何 ACL 类型。
#include "fe/common.hpp"
#include "fe/half.hpp"
#include "fe/startup_trace.hpp"
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_topk.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_scatter_nd_update.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_add_rms_norm.h>
#include <aclnnop/aclnn_swi_glu.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_min_dim.h>
#include <aclnnop/aclnn_apply_rotary_pos_emb_v2.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_sin.h>
#include <atb/atb_infer.h>
#include <algorithm>
#include <bit>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <memory>
#include <numeric>

namespace fe::ascend {
inline void check(int rc,const char* where){if(rc){std::string message=std::string(where)+" 失败，状态="+std::to_string(rc);if(auto detail=aclGetRecentErrMsg();detail&&*detail)message+="；"+std::string(detail);throw std::runtime_error(message);}}
inline int64_t elements(const std::vector<int64_t>& shape){int64_t n=1;for(auto d:shape){require(d>0&&n<=INT64_MAX/d,"Ascend shape 溢出");n*=d;}return n;}
inline size_t bytes(aclDataType dtype){switch(dtype){case ACL_FLOAT:case ACL_INT32:return 4;case ACL_INT64:return 8;case ACL_FLOAT16:return 2;case ACL_INT8:case ACL_BOOL:return 1;default:throw std::runtime_error("未实现的 ACL dtype");}}
inline uint16_t fp16(float value){return float_to_half(value);}
struct Memory {void* data=nullptr;size_t size=0;explicit Memory(size_t n):size(n){check(aclrtMalloc(&data,n,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc");}~Memory(){if(data)aclrtFree(data);}};
struct Tensor {
  std::shared_ptr<Memory> memory;std::vector<int64_t> shape,stride;int64_t offset=0;aclDataType dtype;aclTensor* handle=nullptr;
  // ACLNN 使用逻辑 ND 描述；ATB 单独携带物理 NZ 描述，禁止通过 ND view 解释 NZ 数据。
  aclFormat atb_format=ACL_FORMAT_ND;std::vector<int64_t> atb_shape;
  Tensor(std::shared_ptr<Memory> m,std::vector<int64_t> s,aclDataType d,std::vector<int64_t> st={},int64_t off=0):memory(std::move(m)),shape(std::move(s)),stride(std::move(st)),offset(off),dtype(d){
    if(stride.empty()){stride.resize(shape.size());int64_t acc=1;for(int j=int(shape.size())-1;j>=0;--j){stride[j]=acc;acc*=shape[j];}}
    require(stride.size()==shape.size()&&offset>=0,"非法 Ascend view");int64_t last=offset;for(size_t j=0;j<shape.size();++j){require(shape[j]>0&&stride[j]>=0,"非法 Ascend shape/stride");if(shape[j]>1)require(stride[j]<=(INT64_MAX-last)/(shape[j]-1),"Ascend view stride 溢出");last+=(shape[j]-1)*stride[j];}require(uint64_t(last)<memory->size/bytes(dtype),"Ascend view 超出存储");
    int64_t storage=memory->size/bytes(dtype);handle=aclCreateTensor(shape.data(),shape.size(),dtype,stride.data(),offset,ACL_FORMAT_ND,&storage,1,memory->data);require(handle,"aclCreateTensor 失败");
  }
  ~Tensor(){if(handle)aclDestroyTensor(handle);}
  void* data()const{return static_cast<char*>(memory->data)+offset*bytes(dtype);}
  size_t size()const{return elements(shape)*bytes(dtype);}
};
using T=std::shared_ptr<Tensor>;
inline T tensor(std::vector<int64_t> shape,aclDataType dtype=ACL_FLOAT16){auto n=elements(shape);require(uint64_t(n)<=SIZE_MAX/bytes(dtype),"Ascend tensor 字节数溢出");return std::make_shared<Tensor>(std::make_shared<Memory>(n*bytes(dtype)),std::move(shape),dtype);}
inline void upload(const T& t,const void* data,size_t size){require(size==t->size(),"Ascend 上传长度不匹配");check(aclrtMemcpy(t->data(),t->size(),data,size,ACL_MEMCPY_HOST_TO_DEVICE),"H2D");}
inline std::vector<int64_t> read_ids(const T& t,aclrtStream stream){require(t->dtype==ACL_INT64,"token 读回必须为 int64");check(aclrtSynchronizeStream(stream),"读回同步");std::vector<int64_t> out(elements(t->shape));check(aclrtMemcpy(out.data(),out.size()*8,t->data(),out.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"D2H");return out;}
inline atb::Tensor atb_tensor(const T& t){
  atb::Tensor out;out.desc.dtype=t->dtype;out.desc.format=t->atb_format;const auto& dims=t->atb_shape.empty()?t->shape:t->atb_shape;
  out.desc.shape.dimNum=dims.size();for(size_t i=0;i<dims.size();++i)out.desc.shape.dims[i]=dims[i];out.deviceData=t->data();
  // ND 的连续子视图只暴露有效元素；NZ 权重则保留 16 对齐后的实际存储字节。
  out.dataSize=t->atb_format==ACL_FORMAT_FRACTAL_NZ?t->memory->size-t->offset*bytes(t->dtype):t->size();return out;
}
inline atb::Tensor atb_host(const std::vector<int32_t>& values){atb::Tensor out;out.desc.dtype=ACL_INT32;out.desc.format=ACL_FORMAT_ND;out.desc.shape.dimNum=1;out.desc.shape.dims[0]=values.size();out.hostData=const_cast<int32_t*>(values.data());out.dataSize=values.size()*4;return out;}
struct AtbContext {atb::Context* value=nullptr;AtbContext(){check(atb::CreateContext(&value),"ATB 创建上下文");}~AtbContext(){if(value)atb::DestroyContext(value);}};
struct AtbCall {
  atb::Operation* op=nullptr;std::shared_ptr<AtbContext> owner;atb::VariantPack pack;std::unique_ptr<Memory> workspace;uint64_t size=0;bool ready=false;
  ~AtbCall(){if(op)atb::DestroyOperation(op);}
  template<class P>AtbCall(const P& param,const std::vector<atb::Tensor>& inputs,const std::vector<atb::Tensor>& outputs,std::shared_ptr<AtbContext> context):owner(std::move(context)){
    check(atb::CreateOperation(param,&op),"ATB 创建算子");for(auto& t:inputs)pack.inTensors.push_back(t);for(auto& t:outputs)pack.outTensors.push_back(t);
  }
  void status(int code,const char* phase){if(!code)return;std::string message=std::string("ATB ")+phase+" "+op->GetName();for(size_t i=0;i<pack.inTensors.size();++i){auto& desc=pack.inTensors[i].desc;message+=" input"+std::to_string(i)+"=";for(uint64_t j=0;j<desc.shape.dimNum;++j)message+=std::to_string(desc.shape.dims[j])+",";message+="dtype"+std::to_string(desc.dtype)+"/format"+std::to_string(desc.format);}check(code,message.c_str());}
  void run(aclrtStream stream){auto context=owner->value;if(!ready){check(context->SetExecuteStream(stream),"ATB 设置 stream");status(op->Setup(pack,size,context),"Setup");if(size)workspace=std::make_unique<Memory>(size);ready=true;}status(op->Execute(pack,workspace?static_cast<uint8_t*>(workspace->data):nullptr,size,context),"Execute");}
};
struct Program {
  using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
  struct Op {Execute execute;uint64_t workspace;aclOpExecutor* executor;};
  std::vector<Op> ops;std::vector<T> retained;std::vector<aclScalar*> scalars;std::vector<aclIntArray*> arrays;
  std::unique_ptr<Memory> workspace;uint64_t workspace_size=0;aclmdlRI graph=nullptr;
  std::vector<std::function<void(aclrtStream)>> steps;std::vector<std::shared_ptr<AtbCall>> atb_calls;std::shared_ptr<AtbContext> atb_context;
  Program()=default;Program(const Program&)=delete;Program& operator=(const Program&)=delete;
  ~Program(){if(graph)aclmdlRIDestroy(graph);for(auto& op:ops)aclDestroyAclOpExecutor(op.executor);for(auto p:arrays)aclDestroyIntArray(p);for(auto p:scalars)aclDestroyScalar(p);}
  template<class F,class... Args>void add(F first,Execute execute,Args... args){
    Dl_info symbol{};dladdr(reinterpret_cast<void*>(execute),&symbol);std::string name=symbol.dli_sname?symbol.dli_sname:"未知算子";
    uint64_t size=0;aclOpExecutor* executor=nullptr;check(first(args...,&size,&executor),("ACLNN 一阶段 "+name).c_str());
    try{check(aclSetAclOpExecutorRepeatable(executor),("executor 复用 "+name+" #"+std::to_string(ops.size())).c_str());ops.push_back({execute,size,executor});workspace_size=std::max(workspace_size,size);steps.push_back([this,index=ops.size()-1](aclrtStream stream){auto& op=ops[index];check(op.execute(workspace?workspace->data:nullptr,op.workspace,op.executor,stream),"ACLNN 执行");});}catch(...){aclDestroyAclOpExecutor(executor);throw;}
    if(startup_trace_enabled()&&ops.size()%1000==0)startup_trace("ACLNN 执行计划算子数="+std::to_string(ops.size()));
  }
  template<class P>void atb(const P& param,const std::vector<T>& inputs,const std::vector<T>& outputs,const std::shared_ptr<std::vector<int32_t>>& host={}){
    std::vector<atb::Tensor> in,out;for(auto& t:inputs){keep(t);in.push_back(atb_tensor(t));}if(host)in.push_back(atb_host(*host));for(auto& t:outputs){keep(t);out.push_back(atb_tensor(t));}if(!atb_context)atb_context=std::make_shared<AtbContext>();auto call=std::make_shared<AtbCall>(param,in,out,atb_context);atb_calls.push_back(call);steps.push_back([call,host](aclrtStream stream){call->run(stream);});
  }
  T nz(const T& input){
    require(input->shape.size()==2&&input->dtype==ACL_FLOAT16,"NZ 权重要求 FP16 矩阵");int64_t n=input->shape[0],k=input->shape[1];auto physical=std::vector<int64_t>{(k+15)/16,(n+15)/16,16,16};auto memory=std::make_shared<Memory>(elements(physical)*2);auto output=keep(std::make_shared<Tensor>(memory,input->shape,input->dtype));output->atb_format=ACL_FORMAT_FRACTAL_NZ;output->atb_shape=physical;atb::infer::TransdataParam param;param.transdataType=atb::infer::TransdataParam::ND_TO_FRACTAL_NZ;atb(param,{dense(input)},{output});return output;
  }
  T keep(T value){retained.push_back(value);return value;}
  T out(std::vector<int64_t> shape,aclDataType dtype=ACL_FLOAT16){return keep(tensor(std::move(shape),dtype));}
  bool contiguous(const T& x)const{int64_t step=1;for(int j=int(x->shape.size())-1;j>=0;--j){if(x->shape[j]>1&&x->stride[j]!=step)return false;step*=x->shape[j];}return true;}
  void copy(const T& dst,const T& src){require(dst->shape==src->shape&&dst->dtype==src->dtype,"ACL copy 形状/dtype 不一致");add(aclnnInplaceCopyGetWorkspaceSize,aclnnInplaceCopy,dst->handle,src->handle);}
  T dense(const T& x){if(contiguous(x))return x;auto y=out(x->shape,x->dtype);copy(y,x);return y;}
  T view(const T& x,std::vector<int64_t> shape,std::vector<int64_t> stride={},int64_t offset=0){
    auto source=x;if(stride.empty()){require(!offset&&elements(shape)==elements(x->shape),"reshape 元素数不一致");source=dense(x);}
    return keep(std::make_shared<Tensor>(source->memory,std::move(shape),source->dtype,std::move(stride),source->offset+offset));
  }
  T transpose(const T& x,int a,int b){auto shape=x->shape,stride=x->stride;std::swap(shape[a],shape[b]);std::swap(stride[a],stride[b]);return view(x,shape,stride);}
  T narrow(const T& x,int dim,int64_t begin,int64_t count){auto shape=x->shape;require(begin>=0&&count>0&&begin+count<=shape.at(dim),"Ascend slice 越界");shape[dim]=count;return view(x,shape,x->stride,begin*x->stride[dim]);}
  aclScalar* scalar(float value){auto p=aclCreateScalar(&value,ACL_FLOAT);require(p,"scalar 创建失败");scalars.push_back(p);return p;}
  aclScalar* integer_scalar(int64_t value){auto p=aclCreateScalar(&value,ACL_INT64);require(p,"int scalar 创建失败");scalars.push_back(p);return p;}
  T cast(const T& x,aclDataType dtype){if(x->dtype==dtype)return x;auto y=out(x->shape,dtype);add(aclnnCastGetWorkspaceSize,aclnnCast,x->handle,dtype,y->handle);return y;}
  T cat(const std::vector<T>& inputs,int dim){auto shape=inputs.at(0)->shape;shape[dim]=0;for(auto& x:inputs)shape[dim]+=x->shape[dim];auto y=out(shape,inputs[0]->dtype);int64_t offset=0;for(auto& x:inputs){auto dst=narrow(y,dim,offset,x->shape[dim]);add(aclnnInplaceCopyGetWorkspaceSize,aclnnInplaceCopy,dst->handle,x->handle);offset+=x->shape[dim];}return y;}
  T matmul(const T& a,const T& b){
    require(a->dtype==b->dtype&&a->shape.size()==b->shape.size()&&(a->shape.size()==2||a->shape.size()==3)&&a->shape.back()==b->shape[b->shape.size()-2],"参考 matmul 要求同 dtype 的 2D/3D 输入");
    auto shape=a->shape;shape.back()=b->shape.back();if(shape.size()==3)require(a->shape[0]==b->shape[0],"BatchMatMul 不隐式广播 batch");auto y=out(shape,a->dtype);
    // 310P 的 FLOAT32 cubeMathType=0 不可用；保守保持 FP16 GEMM，GDN 归约全用 FP32 向量运算。
    int8_t mode=a->dtype==ACL_FLOAT?1:0;if(shape.size()==3)add(aclnnBatchMatMulGetWorkspaceSize,aclnnBatchMatMul,a->handle,b->handle,y->handle,mode);else add(aclnnMatmulGetWorkspaceSize,aclnnMatmul,a->handle,b->handle,y->handle,mode);return y;
  }
  T linear(const T& x,const T& weight){if(weight->atb_format==ACL_FORMAT_FRACTAL_NZ){auto input=dense(x),out= this->out({x->shape[0],weight->shape[0]});auto logical=keep(std::make_shared<Tensor>(weight->memory,weight->shape,weight->dtype));logical->atb_format=ACL_FORMAT_FRACTAL_NZ;atb::infer::LinearParam param;param.hasBias=false;atb(param,{input,logical},{out});return out;}return matmul(x,transpose(weight,weight->shape.size()-2,weight->shape.size()-1));}
  T fast_norm(const T& x,const T& weight,double eps){auto input=dense(x);auto y=out(x->shape);auto shape=x->shape;shape.back()=1;auto rstd=out(shape,ACL_FLOAT);add(aclnnRmsNormGetWorkspaceSize,aclnnRmsNorm,input->handle,weight->handle,eps,y->handle,rstd->handle);return y;}
  std::pair<T,T> add_norm(const T& x,const T& residual,const T& weight,double eps){auto a=dense(x),b=dense(residual),y=out(x->shape),sum=out(x->shape);auto shape=x->shape;shape.back()=1;auto rstd=out(shape,ACL_FLOAT);add(aclnnAddRmsNormGetWorkspaceSize,aclnnAddRmsNorm,a->handle,b->handle,weight->handle,eps,y->handle,rstd->handle,sum->handle);return {y,sum};}
  T swiglu(const T& input){auto x=dense(input);auto shape=x->shape;shape.back()/=2;auto y=out(shape);add(aclnnSwiGluGetWorkspaceSize,aclnnSwiGlu,x->handle,int64_t(-1),y->handle);return y;}
  T compare(const T& a,const T& b,bool equal){auto shape=a->shape;for(size_t i=0;i<shape.size();++i)shape[i]=std::max(shape[i],b->shape.at(i));auto y=out(shape,ACL_BOOL);if(equal)add(aclnnEqTensorGetWorkspaceSize,aclnnEqTensor,a->handle,b->handle,y->handle);else add(aclnnLtTensorGetWorkspaceSize,aclnnLtTensor,a->handle,b->handle,y->handle);return y;}
  T minimum(const T& x){auto shape=x->shape;shape.back()=1;auto y=out(shape,ACL_FLOAT),indices=out(shape,ACL_INT64);add(aclnnMinDimGetWorkspaceSize,aclnnMinDim,x->handle,int64_t(-1),true,y->handle,indices->handle);return y;}
  T binary(const T& a,const T& b,char kind){require(a->dtype==b->dtype,"ACL binary 禁止隐式 dtype 提升");std::vector<int64_t> shape(std::max(a->shape.size(),b->shape.size()),1);for(size_t i=0;i<shape.size();++i){int64_t x=i<a->shape.size()?a->shape[a->shape.size()-1-i]:1,y=i<b->shape.size()?b->shape[b->shape.size()-1-i]:1;require(x==y||x==1||y==1,"ACL binary 广播不兼容");shape[shape.size()-1-i]=std::max(x,y);}auto y=out(shape,a->dtype);
    bool integer=a->dtype==ACL_INT64||a->dtype==ACL_INT32||a->dtype==ACL_INT8;
    if(kind=='+')add(aclnnAddGetWorkspaceSize,aclnnAdd,a->handle,b->handle,integer?integer_scalar(1):scalar(1),y->handle);else if(kind=='-')add(aclnnSubGetWorkspaceSize,aclnnSub,a->handle,b->handle,integer?integer_scalar(1):scalar(1),y->handle);else if(kind=='*')add(aclnnMulGetWorkspaceSize,aclnnMul,a->handle,b->handle,y->handle);else if(kind=='/')add(aclnnDivGetWorkspaceSize,aclnnDiv,a->handle,b->handle,y->handle);else throw std::runtime_error("未知 ACL binary");return y;
  }
  T constant(float value,aclDataType dtype){auto x=out({1},dtype);if(dtype==ACL_FLOAT)upload(x,&value,4);else if(dtype==ACL_FLOAT16){auto half=fp16(value);upload(x,&half,2);}else if(dtype==ACL_INT64){auto integer=int64_t(value);upload(x,&integer,8);}else throw std::runtime_error("未实现的 constant dtype");return x;}
  T indices(const std::vector<int64_t>& values){auto x=out({int64_t(values.size())},ACL_INT64);upload(x,values.data(),values.size()*8);return x;}
  T gather(const T& x,const T& ids,int dim){auto shape=x->shape;shape[dim]=elements(ids->shape);auto y=out(shape,x->dtype);add(aclnnGatherV2GetWorkspaceSize,aclnnGatherV2,x->handle,int64_t(dim),ids->handle,y->handle);return y;}
  void scatter(const T& cache,const T& ids,const T& values){
    auto source=dense(values);auto count=elements(ids->shape);
    require(source->dtype==cache->dtype&&source->shape.size()==cache->shape.size()&&source->shape[0]==count,"KV scatter 形状/dtype 不一致");
    for(size_t dim=1;dim<cache->shape.size();++dim)require(source->shape[dim]==cache->shape[dim],"KV scatter 行形状不一致");
    // 当前 310P SDK 的 InplaceIndexCopy 不支持 executor 复用；ND 更新保留动态槽位语义。
    auto indices=view(ids,{count,1});add(aclnnScatterNdUpdateGetWorkspaceSize,aclnnScatterNdUpdate,cache->handle,indices->handle,source->handle);
  }
  T norm(const T& x,const T& weight,double eps){auto xf=cast(x,ACL_FLOAT),w=cast(weight,ACL_FLOAT),y=out(x->shape,ACL_FLOAT);auto shape=x->shape;shape.back()=1;auto rstd=out(shape,ACL_FLOAT);add(aclnnRmsNormGetWorkspaceSize,aclnnRmsNorm,xf->handle,w->handle,eps,y->handle,rstd->handle);return cast(y,ACL_FLOAT16);}
  T unary(const T& x,const std::string& op){auto y=out(x->shape,x->dtype);if(op=="exp")add(aclnnExpGetWorkspaceSize,aclnnExp,x->handle,y->handle);else if(op=="rsqrt")add(aclnnRsqrtGetWorkspaceSize,aclnnRsqrt,x->handle,y->handle);else if(op=="sigmoid")add(aclnnSigmoidGetWorkspaceSize,aclnnSigmoid,x->handle,y->handle);else if(op=="silu")add(aclnnSiluGetWorkspaceSize,aclnnSilu,x->handle,y->handle);else if(op=="softplus")add(aclnnSoftplusGetWorkspaceSize,aclnnSoftplus,x->handle,scalar(1),scalar(20),y->handle);else throw std::runtime_error("未知 ACL unary："+op);return y;}
  T scale(const T& x,float n){return binary(x,constant(n,x->dtype),'*');}
  T l2(const T& value,double eps){auto x=cast(value,ACL_FLOAT);auto denominator=unary(binary(sum(binary(x,x,'*'),int(x->shape.size())-1,true),constant(eps,ACL_FLOAT),'+'),"rsqrt");return binary(x,denominator,'*');}
  T silu_mul(const T& gu){int64_t width=gu->shape.back()/2;auto input=cast(gu,ACL_FLOAT);auto a=narrow(input,input->shape.size()-1,0,width),b=narrow(input,input->shape.size()-1,width,width),activated=out(a->shape,ACL_FLOAT);add(aclnnSiluGetWorkspaceSize,aclnnSilu,a->handle,activated->handle);return cast(binary(activated,b,'*'),ACL_FLOAT16);}
  T softmax(const T& input){auto x=cast(input,ACL_FLOAT),y=out(x->shape,ACL_FLOAT);add(aclnnSoftmaxGetWorkspaceSize,aclnnSoftmax,x->handle,int64_t(-1),y->handle);return y;}
  T sum(const T& x,int dim,bool keepdim){auto axis=int64_t(dim);auto dims=aclCreateIntArray(&axis,1);require(dims,"axis 创建失败");arrays.push_back(dims);auto shape=x->shape;if(keepdim)shape[dim]=1;else shape.erase(shape.begin()+dim);auto y=out(shape,x->dtype);add(aclnnReduceSumGetWorkspaceSize,aclnnReduceSum,x->handle,dims,keepdim,x->dtype,y->handle);return y;}
  std::pair<T,T> topk(const T& x,int64_t k){auto shape=x->shape;shape.back()=k;auto values=out(shape,x->dtype),ids=out(shape,ACL_INT64);add(aclnnTopkGetWorkspaceSize,aclnnTopk,x->handle,k,int64_t(-1),true,true,values->handle,ids->handle);return {values,ids};}
  T argmax(const T& x){auto shape=x->shape;shape.back()=1;auto y=out(shape,ACL_INT64);add(aclnnArgMaxGetWorkspaceSize,aclnnArgMax,x->handle,int64_t(-1),true,y->handle);return y;}
  void finish(){if(workspace_size)workspace=std::make_unique<Memory>(workspace_size);}
  void eager(aclrtStream stream){for(auto& step:steps)step(stream);}
  void capture(aclrtStream stream){eager(stream);check(aclrtSynchronizeStream(stream),"Graph 预热");check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"ACL Graph begin");try{eager(stream);check(aclmdlRICaptureEnd(stream,&graph),"ACL Graph end");}catch(...){aclmdlRI abandoned=nullptr;aclmdlRICaptureEnd(stream,&abandoned);if(abandoned)aclmdlRIDestroy(abandoned);throw;}}
  void run(aclrtStream stream){if(graph)check(aclmdlRIExecuteAsync(graph,stream),"ACL Graph replay");else eager(stream);}
};
}
