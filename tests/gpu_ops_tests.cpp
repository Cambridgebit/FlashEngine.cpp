// 实际设备上的接受内核与 Graph 数据更新测试；不替代模型端到端差分。
#include "fe/common.hpp"
#include "fe/bundle.hpp"
#include "fe/native_cuda.hpp"
#include "fe/weight_loader.hpp"
#include "fe/prefix_state.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc,char** argv){try{
  c10::InferenceMode inference;at::cuda::set_device(0);
  auto side=at::cuda::getStreamFromPool();at::cuda::CUDAStreamGuard guard(side);
  auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  if(argc==3&&std::string(argv[1])=="--quark"){
    auto source=fe::parse_safetensors(std::make_shared<fe::MappedFile>(argv[2]));
    int checks=0;int64_t elements=0;
    for(auto& [name,w]:source)if(w.dtype=="I8"&&name.ends_with(".weight")){
      auto prefix=name.substr(0,name.size()-7);auto& scale=source.at(prefix+".weight_scale");auto& zero=source.at(prefix+".weight_zero_point");
      fe::require(scale.dtype=="F32"&&zero.dtype=="I8","测试夹具必须包含 FP32 scale 和非零 int8 zero point");
      auto q=static_cast<const int8_t*>(w.data()),z=static_cast<const int8_t*>(zero.data());auto scales=static_cast<const float*>(scale.data());
      std::vector<float> reference(w.shape[0]*w.shape[1]);for(int64_t row=0;row<w.shape[0];++row)for(int64_t col=0;col<w.shape[1];++col)reference[row*w.shape[1]+col]=(int(q[row*w.shape[1]+col])-int(z[row]))*scales[row];
      auto expected=at::from_blob(reference.data(),w.shape,at::TensorOptions().dtype(at::kFloat));
      for(auto dtype:{at::kHalf,at::kBFloat16}){auto actual=fe::load_gpu_weight(source,name,at::TensorOptions().device(at::kCUDA).dtype(dtype));fe::require(at::equal(actual.to(at::kCPU),expected.to(dtype)),"Quark 逐通道反量化不一致");++checks;elements+=actual.numel();}
    }
    fe::require(checks>0,"量化夹具为空");std::cout<<fe::Json{{"passed",true},{"checks",checks},{"elements",elements},{"int8_compute",false}}.dump()<<std::endl;return 0;
  }
  if(argc==2){
    auto source=fe::parse_safetensors(std::make_shared<fe::MappedFile>(argv[1]));
    auto load=[&](const std::string& name){auto& value=source.at(name);fe::require(value.dtype=="F16"||value.dtype=="BF16","算子参考必须为 fp16/bf16");auto dtype=value.dtype=="F16"?at::kHalf:at::kBFloat16;return at::from_blob(const_cast<void*>(value.data()),value.shape,at::TensorOptions().dtype(dtype)).to(at::kCUDA);};
    fe::Json cases=fe::Json::array();bool passed=true;
    for(auto& [name,weight]:source)if(name.ends_with(".x")){
      auto prefix=name.substr(0,name.size()-2);auto x=load(name),w=load(prefix+".weight"),residual=load(prefix+".residual"),gu=load(prefix+".gu");
      auto norm=at::empty_like(x),added=at::empty_like(x),sum=at::empty_like(x),silu=at::empty_like(x);int rows=x.size(0),width=x.size(1);bool bf16=weight.dtype=="BF16";
      fe::cuda::norm(x.data_ptr(),nullptr,w.data_ptr(),norm.data_ptr(),nullptr,rows,width,1e-6,bf16,side.stream());
      fe::cuda::norm(x.data_ptr(),residual.data_ptr(),w.data_ptr(),added.data_ptr(),sum.data_ptr(),rows,width,1e-6,bf16,side.stream());
      fe::cuda::silu_mul(gu.data_ptr(),silu.data_ptr(),rows,width,bf16,side.stream());
      for(auto& [operation,actual]:std::vector<std::pair<std::string,at::Tensor>>{{"norm",norm},{"add_norm",added},{"new_residual",sum},{"silu",silu}}){
        auto expected=load(prefix+"."+operation);int64_t different=(actual!=expected).sum().item<int64_t>();
        cases.push_back({{"case",prefix+"."+operation},{"different",different},{"elements",actual.numel()},{"max_abs_error",(actual.to(at::kFloat)-expected.to(at::kFloat)).abs().max().item<float>()}});passed=passed&&different==0;
      }
    }
    fe::require(!cases.empty(),"算子参考没有样例");std::cout<<fe::Json{{"passed",passed},{"cases",cases}}.dump()<<std::endl;return passed?0:2;
  }
  fe::require(argc==1,"用法：fe-gpu-ops-tests [离线算子 oracle.safetensors]");
  {
    fe::PrefixStateCache cache(64);auto original=at::arange(8,options.dtype(at::kFloat));cache.put({1},{original});original.zero_();
    fe::require(at::equal(cache.get({1})[0],at::arange(8,at::TensorOptions().dtype(at::kFloat))),"前缀快照被后续设备写入污染");
    cache.put({2},{original});cache.get({1});cache.put({3},{original});fe::require(cache.has({1})&&!cache.has({2})&&cache.has({3})&&cache.snapshot()["bytes"]==64,"前缀 LRU/预算错误");
    cache.erase({1});fe::require(cache.snapshot()["bytes"]==32,"前缀快照回收错误");fe::PrefixStateCache disabled(0);disabled.put({1},{original});fe::require(!disabled.has({1}),"零预算不得伪造状态命中");
  }
  int checks=0,replays=0;
  for(int batch:{1,2,4,8})for(int k=1;k<=15;++k){
    auto draft=at::zeros({batch,k},options),verify=at::zeros({batch,k+1},options);
    auto output=at::full({batch*(k+2)+16},-999,options);
    auto execute=[&]{fe::cuda::accept(draft.data_ptr<int64_t>(),verify.data_ptr<int64_t>(),output.data_ptr<int64_t>()+8,batch,k,side.stream());};
    execute();at::cuda::device_synchronize();
    at::cuda::CUDAGraph graph;graph.capture_begin();execute();graph.capture_end();
    for(int first=0;first<=k;++first){
      std::vector<int64_t> d(batch*k),v(batch*(k+1));
      for(int b=0;b<batch;++b){int accepted=(first+b)%(k+1);for(int j=0;j<k;++j)d[b*k+j]=v[b*(k+1)+j]=10+j;v[b*(k+1)+accepted]=99;}
      draft.copy_(at::from_blob(d.data(),{batch,k},at::TensorOptions().dtype(at::kLong)));
      verify.copy_(at::from_blob(v.data(),{batch,k+1},at::TensorOptions().dtype(at::kLong)));
      for(bool captured:{false,true}){
        output.fill_(-999);if(captured){graph.replay();++replays;}else execute();
        auto cpu=output.to(at::kCPU).contiguous();auto* ids=cpu.data_ptr<int64_t>();
        for(int j=0;j<8;++j)fe::require(ids[j]==-999&&ids[8+batch*(k+2)+j]==-999,"接受内核越界写");
        for(int b=0;b<batch;++b){int accepted=(first+b)%(k+1);auto* row=ids+8+b*(k+2);
          fe::require(row[k+1]==accepted+1&&row[accepted]==99,"接受长度或 bonus 错误");
          for(int j=0;j<accepted;++j)fe::require(row[j]==10+j,"接受前缀错误");
          ++checks;
        }
      }
    }
  }
  std::cout<<fe::Json{{"passed",true},{"acceptance_checks",checks},{"graph_replays",replays},{"batch_sizes",{1,2,4,8}},{"k_range",{1,15}}}.dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
