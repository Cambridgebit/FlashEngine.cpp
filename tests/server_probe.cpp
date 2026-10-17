// 仅测试的主机模型：让真实 Beast 服务可在 ASan 下独立检查，不进入部署程序。
#include "fe/server.hpp"
#include <chrono>
#include <thread>
#include <iostream>

class ProbeModel final:public fe::DeviceModel {
public:
  fe::ModelOutput run(const fe::Batch& batch)override{
    std::this_thread::sleep_for(std::chrono::milliseconds(2));fe::ModelOutput out;
    for(auto& s:batch.sequences)out.tokens.push_back(batch.prefill&&s->cached+s->scheduled<s->tokens.size()?std::vector<int64_t>{}:std::vector<int64_t>{7});return out;
  }
  void drop(uint64_t)override{}
  fe::Json snapshot()const override{return {{"backend","test-only"}};}
};
int main(int argc,char** argv){try{
  fe::require(argc>=3&&std::string(argv[1])=="serve","测试服务需要 serve BUNDLE");fe::Bundle bundle(argv[2]);fe::Tokenizer tokenizer(bundle.root);fe::Config config;fe::ServerConfig server;
  for(int i=3;i<argc;++i){std::string flag=argv[i];auto value=[&](){fe::require(i+1<argc,"参数缺少值");return std::string(argv[++i]);};
    if(flag=="--host")server.host=value();else if(flag=="--port")server.port=std::stoi(value());else if(flag=="--served-model-name")server.model=value();else if(flag=="--api-key")server.api_key=value();else if(flag=="--max-model-len")config.max_model_len=std::stoll(value());else if(flag=="--max-num-seqs")config.max_num_seqs=std::stoll(value());else if(flag=="--kv-blocks")config.num_kvcache_blocks=std::stoll(value());else if(flag=="--max-num-batched-tokens")config.max_num_batched_tokens=std::stoll(value());else if(flag=="--max-num-partial-prefills")config.max_num_partial_prefills=std::stoll(value());else if(flag=="--no-think")server.thinking=false;else if(flag!="--graph"&&flag!="--eager")throw fe::InputError("未知测试参数");
  }
  config.validate();fe::Engine engine(config,std::make_unique<ProbeModel>());return fe::serve(engine,tokenizer,server,bundle.config["target"]["vocab_size"]);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
