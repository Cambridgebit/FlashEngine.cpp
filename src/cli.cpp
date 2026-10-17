#include "fe/model.hpp"
#include "fe/server.hpp"
#include "fe/profiles.hpp"
#include "fe/benchmark.hpp"
#include <iostream>

namespace fe {
int run_device_cli(const std::string& command,Bundle& bundle,Tokenizer& tokenizer,int argc,char** argv,const std::string& backend,const DeviceFactory& factory,const std::string& default_dtype){
  if(command=="benchmark")return run_benchmark_cli(bundle,argc,argv,backend,factory);
  require(command=="generate"||command=="serve"||command=="replay","未知命令");Config config;config.backend=backend;config.dtype=default_dtype;ServerConfig server;std::string prompt="你好，请用一句话介绍自己。",oracle;int64_t limit=32;bool chat=false,teacher=false;
  for(int i=3;i<argc;++i){std::string flag=argv[i];auto value=[&](){require(i+1<argc,"选项缺少值");return std::string(argv[++i]);};
    if(flag=="--backend")config.backend=value();
    else if(flag=="--logits")config.trace_logits=true;
    else if(flag=="--prefix-cache")config.enable_prefix_cache=true;
    else if(flag=="--no-prefix-cache")config.enable_prefix_cache=false;
    else if(flag=="--enable-chunked-prefill")config.enable_chunked_prefill=true;
    else if(flag=="--disable-chunked-prefill")config.enable_chunked_prefill=false;
    else if(flag=="--recurrent-prefix-cache-bytes")config.recurrent_prefix_cache_bytes=parse_integer(value());
    else if(flag=="--teacher-force")teacher=true;
    else if(flag=="--prompt")prompt=value();
    else if(flag=="--oracle")oracle=value();
    else if(flag=="--max-tokens")limit=parse_integer(value());
    else if(flag=="--max-model-len")config.max_model_len=parse_integer(value());
    else if(flag=="--max-num-seqs")config.max_num_seqs=parse_integer(value());
    else if(flag=="--max-num-batched-tokens")config.max_num_batched_tokens=parse_integer(value());
    else if(flag=="--max-num-partial-prefills")config.max_num_partial_prefills=parse_integer(value());
    else if(flag=="--kv-blocks")config.num_kvcache_blocks=parse_integer(value());
    else if(flag=="--kv-block-size")config.kvcache_block_size=parse_integer(value());
    else if(flag=="--dtype")config.dtype=value();
    else if(flag=="--host")server.host=value();
    else if(flag=="--port"){auto n=parse_integer(value());require(n>0&&n<=65535,"端口必须为 1..65535");server.port=n;}
    else if(flag=="--api-key")server.api_key=value();
    else if(flag=="--served-model-name")server.model=value();
    else if(flag=="--reserve-output"){auto n=parse_integer(value());require(n>=0&&n<=INT32_MAX,"非法输出预留");server.reserve_output=n;}
    else if(flag=="--no-think")server.thinking=false;
    else if(flag=="--graph")config.enforce_eager=false;
    else if(flag=="--eager")config.enforce_eager=true;
    else if(flag=="--chat")chat=true;
    else if(flag=="--production")config.production=true;
    else throw InputError("未知参数："+flag);
  }
  config.eos=bundle.tokenizer_runtime.at("eos_token_id");if(!bundle.config.at("draft").is_null())config.spec_num_tokens=bundle.config["draft"]["num_tokens"];
  config.validate();require(config.backend==backend,"此部署物未编译请求的后端："+config.backend+"；当前后端="+backend);
  std::cerr<<"[INFO ] resolved_config="<<config.json().dump()<<std::endl;
  auto model=factory(bundle,config);
  if(command=="serve"){Engine engine(config,std::move(model));return serve(engine,tokenizer,server,bundle.config["target"]["vocab_size"]);}
  if(command=="replay"){
    require(!teacher||config.spec_num_tokens==0,"teacher-force 仅用于普通目标模型逐 token 诊断");
    require(!oracle.empty(),"replay 需要 --oracle");auto golden=read_json(oracle);require(compatible_contract(golden.at("contract_version")),"oracle 契约版本错误");require(golden.at("oracle").at("bundle_sha256")==sha256(bundle.root/"bundle.json"),"oracle 与模型 bundle 不匹配");
    require(golden.at("cases").is_array()&&!golden.at("cases").empty(),"oracle 必须包含非空样例");
    auto environment=golden.at("oracle").value("environment",Json::object());
    if(environment.contains("dtype"))require(environment["dtype"]=="torch."+config.dtype||environment["dtype"]==config.dtype,"oracle 与原生计算 dtype 不一致");
    Json results=Json::array();bool passed=true;int index=0;
    for(auto& item:golden.at("cases")){
      require(item.at("sampling").at("temperature")==0,"原生 replay 仅支持 greedy oracle");
      Scheduler scheduler(config);bind_model(scheduler,*model);auto seq=std::make_shared<Sequence>(++index,item.at("prompt_token_ids").get<std::vector<int64_t>>(),item.at("sampling").at("max_tokens"));seq->ignore_eos=item.at("sampling").value("ignore_eos",false);scheduler.add(seq);Json rounds=Json::array(),samples=Json::array();std::vector<int64_t> observed;
      while(!scheduler.empty()){auto batch=scheduler.schedule();auto output=model->run(batch);for(size_t i=0;i<batch.sequences.size();++i){auto s=batch.sequences[i];if(batch.prefill)s->cached+=s->scheduled;auto committed=output.tokens.at(i);if(teacher&&!committed.empty()){require(committed.size()==1,"teacher-force 只允许单 token 提交");observed.push_back(committed[0]);committed[0]=item.at("output_token_ids").at(s->tokens.size()-s->prompt_size);}scheduler.commit(s,committed);if(s->status!=Status::finished&&!committed.empty())s->cached=s->tokens.size()-1;s->scheduled=0;}for(auto round:output.rounds){round.erase("request_id");rounds.push_back(round);}for(auto sample:output.samples)samples.push_back(sample);}
      std::vector<int64_t> generated=teacher?observed:std::vector<int64_t>(seq->tokens.begin()+seq->prompt_size,seq->tokens.end());bool ok=generated==item.at("output_token_ids").get<std::vector<int64_t>>()&&seq->finish_reason==item.at("finish_reason");if(!item.value("rounds",Json::array()).empty())ok=ok&&nlohmann::json(rounds)==nlohmann::json(item.at("rounds"));passed=passed&&ok;results.push_back({{"case",index},{"passed",ok},{"teacher_forced",teacher},{"token_ids",generated},{"rounds",rounds},{"samples",samples}});
    }
    std::cout<<Json{{"passed",passed},{"cases",results},{"runtime",model->snapshot()}}.dump()<<std::endl;return passed?0:2;
  }
  auto ids=tokenizer.encode(chat?tokenizer.render(Json::array({{{"role","user"},{"content",prompt}}}),false):prompt);
  Scheduler scheduler(config);bind_model(scheduler,*model);auto seq=std::make_shared<Sequence>(1,ids,limit);scheduler.add(seq);Json rounds=Json::array();
  while(!scheduler.empty()){
    auto batch=scheduler.schedule();if(batch.sequences.empty())break;auto result=model->run(batch);
    for(size_t i=0;i<batch.sequences.size();++i){auto& s=batch.sequences[i];if(batch.prefill)s->cached+=s->scheduled;scheduler.commit(s,result.tokens[i]);if(s->status!=Status::finished&&!result.tokens[i].empty())s->cached=s->tokens.size()-1;s->scheduled=0;}
    for(auto& round:result.rounds)rounds.push_back(round);
  }
  std::vector<int64_t> generated(seq->tokens.begin()+seq->prompt_size,seq->tokens.end());std::cout<<Json{{"text",tokenizer.decode(generated)},{"token_ids",generated},{"prompt_token_ids",ids},{"finish_reason",seq->finish_reason},{"rounds",rounds},{"runtime",model->snapshot()}}.dump()<<std::endl;return 0;
}}
