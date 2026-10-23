#include "fe/benchmark.hpp"
#include "fe/startup_trace.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <map>

namespace fe {
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double unix_time(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}
double percentile(std::vector<double> values,double p){if(values.empty())return 0;std::sort(values.begin(),values.end());double at=(values.size()-1)*p/100;auto low=size_t(at),high=std::min(low+1,values.size()-1);return values[low]+(values[high]-values[low])*(at-low);}
struct Options {int64_t batch=1,warmup=3,repeats=1;bool graph=false,trace=false;std::string workload;};
Config config_from(const Json& workload,const Bundle& bundle,const std::string& backend,bool graph){
  require(workload.at("schema_version")==1,"不支持的 workload 版本");require(workload.at("bundle_sha256")==sha256(bundle.root/"bundle.json"),"workload 与 bundle 不匹配");
  auto& c=workload.at("config");Config out;out.backend=backend;out.enforce_eager=!graph;
  out.dtype=c.at("dtype");out.max_model_len=c.at("max_model_len");out.max_num_seqs=c.at("max_num_seqs");out.max_num_batched_tokens=c.at("max_num_batched_tokens");out.max_num_partial_prefills=c.at("max_num_partial_prefills");out.kvcache_block_size=c.at("kvcache_block_size");out.num_kvcache_blocks=c.at("num_kvcache_blocks");
  out.eos=bundle.tokenizer_runtime.at("eos_token_id");if(!bundle.config.at("draft").is_null())out.spec_num_tokens=bundle.config["draft"]["num_tokens"];
  out.enable_prefix_cache=c.value("enable_prefix_cache",true);
  out.enable_chunked_prefill=c.value("enable_chunked_prefill",true);
  out.recurrent_prefix_cache_bytes=c.value("recurrent_prefix_cache_bytes",out.recurrent_prefix_cache_bytes);
  out.validate();auto requests=workload.at("requests");require(requests.is_array()&&!requests.empty(),"workload 请求为空");
  std::set<std::string> seen;for(auto& r:requests){require(seen.insert(r.at("id").get<std::string>()).second,"workload id 重复");auto ids=r.at("prompt_token_ids").get<std::vector<int64_t>>();int64_t limit=r.at("max_tokens");require(!ids.empty()&&limit>0&&int64_t(ids.size())+limit<=out.max_model_len,"workload 超出上下文，不允许静默裁剪");for(auto id:ids)require(id>=0&&id<bundle.config["target"]["vocab_size"].get<int64_t>(),"workload token 越界");}
  return out;
}
Json group(DeviceModel& model,Scheduler& scheduler,const Config& config,const Json& requests,uint64_t& next,bool trace,int repeat){
  auto preemptions_before=scheduler.preemptions;std::vector<Seq> seqs;std::map<uint64_t,size_t> index;std::vector<double> first(requests.size(),-1),last(requests.size(),-1);std::vector<Json> rounds(requests.size(),Json::array());
  auto start=Clock::now();for(auto& request:requests){auto seq=std::make_shared<Sequence>(++next,request.at("prompt_token_ids").get<std::vector<int64_t>>(),request.at("max_tokens"));seq->ignore_eos=request.at("ignore_eos");index[seq->id]=seqs.size();seqs.push_back(seq);scheduler.add(seq);}
  while(!scheduler.empty()){
    auto batch=scheduler.schedule();auto output=model.run(batch);
    for(size_t j=0;j<batch.sequences.size();++j){auto s=batch.sequences[j];if(batch.prefill)s->cached+=s->scheduled;scheduler.commit(s,output.tokens.at(j));auto row=index.at(s->id);if(first[row]<0&&s->tokens.size()>s->prompt_size)first[row]=seconds(start);if(s->status==Status::finished)last[row]=seconds(start);if(s->status!=Status::finished&&!output.tokens[j].empty())s->cached=s->tokens.size()-1;s->scheduled=0;}
    if(trace)for(auto round:output.rounds){auto id=round.at("request_id").get<uint64_t>();round.erase("request_id");rounds[index.at(id)].push_back(std::move(round));}
  }
  double wall=seconds(start);require(scheduler.blocks.used()==0,"基准结束后 KV 未释放");Json rows=Json::array();
  for(size_t j=0;j<seqs.size();++j){auto& s=*seqs[j];std::vector<int64_t> ids(s.tokens.begin()+s.prompt_size,s.tokens.end());Json row{{"input_id",requests[j]["id"]},{"repeat",repeat},{"prompt_tokens",s.prompt_size},{"token_ids",ids},{"finish_reason",s.finish_reason},{"ttft_seconds",first[j]},{"e2e_seconds",last[j]},{"group_wall_seconds",wall},{"preemptions",s.preemptions}};if(trace)row["rounds"]=rounds[j];rows.push_back(std::move(row));}
  return {{"rows",rows},{"wall_seconds",wall},{"preemptions",scheduler.preemptions-preemptions_before}};
}
}
int run_benchmark_cli(const Bundle& bundle,int argc,char** argv,const std::string& backend,const DeviceFactory& factory){
  Options options;for(int j=3;j<argc;++j){std::string flag=argv[j];auto value=[&](){require(j+1<argc,"基准选项缺少值");return std::string(argv[++j]);};if(flag=="--workload")options.workload=value();else if(flag=="--batch-size")options.batch=parse_integer(value());else if(flag=="--warmup")options.warmup=parse_integer(value());else if(flag=="--repeats")options.repeats=parse_integer(value());else if(flag=="--graph")options.graph=true;else if(flag=="--eager")options.graph=false;else if(flag=="--trace-rounds")options.trace=true;else throw InputError("未知基准参数："+flag);}
  require(!options.workload.empty()&&options.batch>0&&options.warmup>=0&&options.repeats>0,"非法基准配置");auto workload=read_json(options.workload);auto config=config_from(workload,bundle,backend,options.graph);require(options.batch<=config.max_num_seqs,"batch-size 超过 max_num_seqs");
  startup_trace("开始构造设备模型");auto model=factory(bundle,config);startup_trace("设备模型就绪；开始 benchmark 预热");Scheduler scheduler(config);bind_model(scheduler,*model);auto& requests=workload.at("requests");auto warm=Json::array();for(size_t j=0;j<std::min(size_t(options.batch),requests.size());++j)warm.push_back(requests[j]);uint64_t next=0;
  for(int j=0;j<options.warmup;++j)group(*model,scheduler,config,warm,next,false,-1);
  startup_trace("预热完成；开始主跑计时");auto before=model->snapshot();auto start=Clock::now();double begin=unix_time();Json rows=Json::array();double request_wall=0;uint64_t preemptions=0;
  for(int rep=0;rep<options.repeats;++rep)for(size_t offset=0;offset<requests.size();offset+=options.batch){auto part=Json::array();for(size_t j=offset;j<std::min(offset+options.batch,requests.size());++j)part.push_back(requests[j]);auto result=group(*model,scheduler,config,part,next,options.trace,rep);request_wall+=result["wall_seconds"].get<double>();preemptions+=result["preemptions"].get<uint64_t>();for(auto& row:result["rows"])rows.push_back(std::move(row));}
  double wall=seconds(start),end=unix_time();startup_trace("主跑计时结束");std::vector<double> ttft;uint64_t total=0;for(auto& row:rows){total+=row["token_ids"].size();ttft.push_back(row["ttft_seconds"].get<double>()*1000);}
  Json result{{"implementation","cpp"},{"protocol","pretokenized-grouped-engine-v1"},{"workload_sha256",sha256(options.workload)},{"resolved_config",config.json()},{"batch_size",options.batch},{"warmup",options.warmup},{"repeats",options.repeats},{"trace_rounds",options.trace},{"measurement_unix_start",begin},{"measurement_unix_end",end},{"wall_seconds",wall},{"group_wall_seconds_sum",request_wall},{"output_tokens",total},{"output_tps",total/wall},{"ttft_p50_ms",percentile(ttft,50)},{"ttft_p95_ms",percentile(ttft,95)},{"preemptions",preemptions},{"runtime_before",before},{"runtime_after",model->snapshot()},{"rows",rows}};
  result["scheduler"]=scheduler.snapshot();std::cout<<result.dump()<<std::endl;return 0;
}
}
