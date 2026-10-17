#include "fe/bundle.hpp"
#include "fe/tokenizer.hpp"
#include <iostream>
#if defined(FE_HAS_CUDA) || defined(FE_HAS_ASCEND)
#include "fe/model.hpp"
#endif

int main(int argc,char** argv){
  try{
    if(argc==2&&std::string(argv[1])=="--help"){
      std::cout<<R"(用法：flashengine-native 命令 BUNDLE [选项]
命令：verify、tokenize 文本、tokenize-file 样例.json、generate、replay、serve、benchmark
无需模型：--help；capabilities（JSON 编译能力，不代表验收资格）
执行配置：--backend cuda|corex|ascend310p --dtype float16|bfloat16
          --max-model-len N --max-num-seqs N --max-num-batched-tokens N
          --max-num-partial-prefills N --kv-blocks N --kv-block-size N
          --eager（默认）或 --graph；--production 要求 Graph，不授予发布资格
生成：--prompt 文本 --chat --max-tokens N
差分：--oracle 文件 [--logits] [--teacher-force]（teacher 仅普通目标模型诊断）
基准：benchmark BUNDLE --workload 文件 --batch-size N --warmup N --repeats N
      [--graph|--eager] [--trace-rounds]；完整公共配置由 workload 固定
服务：--host 地址 --port N --served-model-name 名称 --api-key 密钥
      --reserve-output N --no-think
限制：TP=1、greedy、无量化/Prefix cache；具体后端以 capabilities 和验收记录为准。
)";return 0;
    }
    if(argc==2&&std::string(argv[1])=="capabilities"){
      std::string backend="host",attention="none";
#ifdef FE_HAS_CUDA
      backend=FE_DEFAULT_BACKEND;
#ifdef FE_REFERENCE_OPS_BUILD
      attention="aten-paged-reference";
#else
      attention=backend=="corex"?"aten-paged-reference":"flashinfer-native";
#endif
#elif defined(FE_HAS_ASCEND)
      backend="ascend310p";attention="aclnn-nd-reference";
#endif
      fe::Json out{{"contract_version",fe::contract_version},{"compiled_backend",backend},{"attention",attention},{"production_qualified",false},{"profiles",{"qwen3-dense-v1","qwen3-moe-v1","dflash1-greedy-v1"}},{"device_execution",backend!="host"},{"graph",backend!="host"},{"tensor_parallel_size",1},{"quantization",false},{"prefix_cache",false},{"dflash_reduced_vocab",false},{"dflash_sliding_attention",false},{"architectures","Linux x86_64/aarch64，必须使用相应架构和厂商 SDK 编译"}};
      out["dtypes"]=backend=="ascend310p"?fe::Json({"float16"}):fe::Json({"float16","bfloat16"});
      std::cout<<out.dump()<<std::endl;return 0;
    }
    fe::require(argc>=3,"用法：flashengine-native 命令 BUNDLE [选项]；查看 --help 或 capabilities");
    std::string command=argv[1];fe::Bundle bundle(argv[2]);fe::Tokenizer tokenizer(bundle.root);
    tokenizer.verify_golden(fe::read_json(bundle.path("golden/tokenizer_chat.json")));
    if(command=="verify"){std::cout<<fe::Json{{"verified",true},{"contract_version",fe::contract_version},{"target_tensors",bundle.target_weights.size()},{"draft_tensors",bundle.draft_weights.size()}}.dump()<<std::endl;return 0;}
    fe::require(!fe::fs::exists(bundle.root/"INCOMPLETE.json"),"bundle 尚未打包完成");
    if(command=="tokenize"){fe::require(argc>=4,"需要输入文本");auto ids=tokenizer.encode(argv[3]);std::cout<<fe::Json{{"token_ids",ids},{"decoded",tokenizer.decode(ids,false)}}.dump()<<std::endl;return 0;}
    if(command=="tokenize-file"){fe::require(argc==4,"需要输入样例 JSON 文件");auto cases=fe::read_json(argv[3]);fe::Json out=fe::Json::array();bool passed=true;for(auto& item:cases){auto ids=tokenizer.encode(item.at("text"));auto decoded=tokenizer.decode(ids,false);bool ok=ids==item.at("token_ids").get<std::vector<int64_t>>()&&decoded==item.at("decoded");passed=passed&&ok;out.push_back({{"passed",ok},{"token_ids",ids},{"decoded",decoded}});}std::cout<<fe::Json{{"passed",passed},{"cases",out}}.dump()<<std::endl;return passed?0:2;}
#ifdef FE_HAS_CUDA
    return fe::run_device_cli(command,bundle,tokenizer,argc,argv,FE_DEFAULT_BACKEND,fe::create_cuda_model);
#elif defined(FE_HAS_ASCEND)
    return fe::run_device_cli(command,bundle,tokenizer,argc,argv,"ascend310p",fe::create_ascend_model,"float16");
#else
    throw std::runtime_error("此构建不含设备后端；请使用 cuda、corex 或 ascend310p preset");
#endif
  }catch(const std::exception& error){std::cerr<<"[ERROR] "<<error.what()<<std::endl;return 1;}
}
