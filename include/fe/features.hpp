#pragma once
#include "fe/common.hpp"

namespace fe {
// 只描述已编入部署物的源码路径；绝不把编译能力当作真机发布资格。
inline Json source_capabilities(const std::string& backend,const std::string& attention){
  require(backend=="host"||backend=="cuda"||backend=="corex"||backend=="ascend310p","未知能力提供方");
  bool device=backend!="host";
  Json profiles=Json::array(),combinations=Json::array();
  if(device){
    profiles={"qwen3-dense-v1","qwen3-moe-v1","qwen35-dense-v1","qwen35-moe-v1","dflash1-greedy-v1","eagle3-greedy-v1","quark-int8-dequant-v1"};
    for(std::string target:{"qwen3","qwen3_moe","qwen3_5","qwen3_5_moe"})combinations.push_back({{"target",target},{"spec_methods",target.starts_with("qwen3_5")?Json({"none","dflash"}):Json({"none","dflash","eagle3"})},{"prefix_cache",true},{"graph",true}});
  }
  return {{"contract_version",contract_version},{"compiled_backend",backend},{"attention",attention},{"implementation_revision","backend-parity-source-2026-10-17"},{"implementation_status","unvalidated-source-changes"},{"production_qualified",false},{"profiles",profiles},{"combinations",combinations},{"device_execution",device},{"graph",device},{"tensor_parallel_size",1},{"dtypes",backend=="ascend310p"?Json({"float16"}):device?Json({"float16","bfloat16"}):Json::array()},{"prefix_cache",device},{"prefix_cache_default",true},{"prefix_cache_targets",device?Json({"qwen3","qwen3_moe","qwen3_5","qwen3_5_moe"}):Json::array()},{"prefix_cache_speculative",device},{"hybrid_speculative",device},{"hybrid_spec_methods",device?Json({"dflash"}):Json::array()},{"quantization",device},{"quantized_compute",false},{"quantization_modes",device?Json({"quark-int8-dequant-v1"}):Json::array()},{"chunked_prefill",device},{"chunked_prefill_default",true},{"continuous_batching",device},{"partial_prefill_concurrency",device},{"preemption_recompute",device},{"thread_safe_cancellation",device},{"host_only_observability",true},{"dflash_reduced_vocab",false},{"dflash_sliding_attention",false},{"architectures","Linux x86_64/aarch64；需原生厂商 SDK 和对应架构依赖"}};
}
}
