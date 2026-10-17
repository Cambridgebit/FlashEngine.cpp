#pragma once
#include "fe/bundle.hpp"
#include <ATen/ATen.h>

namespace fe {
// 映射保持到 to(device) 完成；反量化只发生在加载期，执行期仍为 FP16/BF16。
inline at::Tensor load_gpu_weight(const std::map<std::string,WeightView>& source,const std::string& name,at::TensorOptions options){
  auto raw=[&](const std::string& key){auto& w=source.at(key);auto dtype=w.dtype=="BF16"?at::kBFloat16:w.dtype=="F16"?at::kHalf:w.dtype=="F32"?at::kFloat:w.dtype=="I8"?at::kChar:at::kLong;return at::from_blob(const_cast<void*>(w.data()),w.shape,at::TensorOptions().dtype(dtype));};
  auto value=raw(name);
  if(source.at(name).dtype=="I8"&&name.ends_with(".weight")){
    auto prefix=name.substr(0,name.size()-7);auto scale=raw(prefix+".weight_scale").to(at::kFloat).reshape({-1,1});
    require(at::isfinite(scale).all().item<bool>()&&(scale>0).all().item<bool>(),"int8 scale 必须有限且为正");
    value=value.to(at::kFloat);
    if(source.contains(prefix+".weight_zero_point")){auto zero=raw(prefix+".weight_zero_point").to(at::kFloat).reshape({-1,1});require(at::isfinite(zero).all().item<bool>(),"int8 zero point 非有限");value=value-zero;}
    value=value*scale;
  }
  return value.to(options);
}
}
