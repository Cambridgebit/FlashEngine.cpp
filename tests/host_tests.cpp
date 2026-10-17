#include "fe/scheduler.hpp"
#include "fe/bundle.hpp"
#include "fe/profiles.hpp"
#include "fe/half.hpp"
#include <limits>
#include <iostream>
#include <fstream>
#include <unistd.h>

int main(int argc,char** argv){try{
  int count=0;auto check=[&](bool ok){fe::require(ok,"主机断言失败 #"+std::to_string(count));++count;};
  check(fe::graph_buckets(7)==std::vector<int64_t>({1,2,4,7}));
  check(fe::parse_integer("2048")==2048&&fe::parse_integer("-1")==-1);
  for(auto text:{"","8x"," 8","1.0","9223372036854775808"}){bool failed=false;try{fe::parse_integer(text);}catch(...){failed=true;}check(failed);}
  check(fe::float_to_half(0.f)==0&&fe::float_to_half(-0.f)==0x8000);
  check(fe::float_to_half(1.f)==0x3c00&&fe::float_to_half(65504.f)==0x7bff);
  check(fe::float_to_half(100000.f)==0x7c00&&fe::float_to_half(-100000.f)==0xfc00);
  check(fe::float_to_half(0x1p-24f)==1&&fe::float_to_half(0x1p-25f)==0);
  check(fe::float_to_half(1.f+0x1p-11f)==0x3c00);
  check((fe::float_to_half(std::numeric_limits<float>::quiet_NaN())&0x7fff)>0x7c00);
  fe::Json moe{{"model_type","qwen3_moe"},{"hidden_size",8},{"intermediate_size",16},{"vocab_size",20},{"num_hidden_layers",4},{"num_attention_heads",2},{"num_key_value_heads",1},{"head_dim",4},{"max_position_embeddings",128},{"tie_word_embeddings",false},{"num_experts",4},{"num_experts_per_tok",2},{"moe_intermediate_size",6},{"decoder_sparse_step",2},{"mlp_only_layers",fe::Json::array()},{"norm_topk_prob",true}};
  check(!fe::moe_layer(moe,0)&&fe::moe_layer(moe,1)&&!fe::moe_layer(moe,2)&&fe::moe_layer(moe,3));
  auto shapes=fe::expected_model_weights(moe,false);check(shapes.at("layers.1.mlp.experts.0.gate_proj.weight")==std::vector<int64_t>({6,8}));
  moe["expert_layouts"]={{"1","fused"}};shapes=fe::expected_model_weights(moe,false);check(shapes.at("layers.1.mlp.experts.gate_up_proj.weight")==std::vector<int64_t>({4,12,8}));
  moe["mlp_only_layers"]={1};check(!fe::moe_layer(moe,1)&&fe::moe_layer(moe,3));
  check(fe::accept_greedy({1,2,3},{1,9,8,7})==std::vector<int64_t>({1,9}));
  check(fe::accept_greedy({1,2},{1,2,3})==std::vector<int64_t>({1,2,3}));
  check(fe::accept_greedy({1},{9,8})==std::vector<int64_t>({9}));
  for(int k=1;k<=15;++k)for(int accepted=0;accepted<=k;++accepted){std::vector<int64_t> draft(k,4),verify(k+1,4);verify[accepted]=9;auto result=fe::accept_greedy(draft,verify);check(result.size()==size_t(accepted+1)&&result.back()==9);}
  bool rejected=false;try{fe::parse_json("{\"a\":1,\"a\":2}");}catch(...){rejected=true;}check(rejected);
  if(argc==2){fe::fs::path root=argv[1];auto lock=fe::read_json(root/"contracts.lock");check(lock["contract_version"]==fe::contract_version);for(auto it=lock["files"].begin();it!=lock["files"].end();++it)check(fe::sha256(root/lock.value("directory",std::string("vendor/contracts"))/it.key())==it.value());}
  fe::BlockManager blocks(2,16);fe::Sequence a(1,{1},10),b(2,{2},10);check(blocks.ensure(a,17));check(!blocks.ensure(b,1));check(b.blocks.empty());blocks.release(a);check(blocks.used()==0);check(blocks.ensure(b,1));blocks.release(b);
  fe::Config config;config.max_model_len=64;config.max_num_seqs=2;config.max_num_batched_tokens=3;config.num_kvcache_blocks=4;config.kvcache_block_size=16;fe::Scheduler sched(config);
  auto s=std::make_shared<fe::Sequence>(1,std::vector<int64_t>{1,2,3,4,5},2);sched.add(s);
  auto batch=sched.schedule();check(batch.prefill&&s->scheduled==3);s->cached+=s->scheduled;
  batch=sched.schedule();check(batch.prefill&&s->scheduled==2);s->cached+=s->scheduled;sched.commit(s,{9});
  batch=sched.schedule();check(!batch.prefill&&batch.sequences.size()==1);sched.commit(s,{10});check(sched.empty()&&sched.blocks.used()==0&&s->finish_reason=="length");
  auto cancel=std::make_shared<fe::Sequence>(2,std::vector<int64_t>{1},3);sched.add(cancel);cancel->cancelled->store(true);check(sched.reap_cancelled().size()==1&&sched.empty());
  auto capped=std::make_shared<fe::Sequence>(3,std::vector<int64_t>(63,1),4);fe::Config c=config;c.num_kvcache_blocks=5;fe::Scheduler cap(c);cap.add(capped);cap.commit(capped,{2,3,4});check(capped->tokens.size()==64&&capped->finish_reason=="length");
  // 用极小 KV 池强制抢占，确保请求不会因恢复而丢输出或永久饥饿。
  c=config;c.max_num_seqs=4;c.max_num_batched_tokens=7;c.max_num_partial_prefills=2;fe::Scheduler pressure(c);std::vector<fe::Seq> requests;
  for(int id=0;id<8;++id){auto seq=std::make_shared<fe::Sequence>(10+id,std::vector<int64_t>(33,1),3);requests.push_back(seq);pressure.add(seq);}
  int steps=0;while(!pressure.empty()&&++steps<2000){auto batch=pressure.schedule();for(auto& seq:batch.sequences){if(batch.prefill)seq->cached+=seq->scheduled;bool complete=!batch.prefill||seq->cached==seq->tokens.size();if(complete)pressure.commit(seq,{7});if(complete&&seq->status!=fe::Status::finished)seq->cached=seq->tokens.size()-1;seq->scheduled=0;}}
  check(pressure.empty()&&pressure.preemptions>0&&pressure.blocks.used()==0);for(auto& seq:requests)check(seq->tokens.size()==36&&seq->finish_reason=="length");
  // 正常和恶意 safetensors 都经实际 mmap 路径测试。
  auto path=fe::fs::temp_directory_path()/("fe-host-"+std::to_string(getpid())+".safetensors");
  auto write=[&](int end){auto header=fe::Json{{"w",{{"dtype","F16"},{"shape",{2}},{"data_offsets",{0,end}}}}}.dump();std::ofstream out(path,std::ios::binary);uint64_t length=header.size();out.write(reinterpret_cast<const char*>(&length),8);out<<header;out.write("abcd",4);};
  write(4);{auto weights=fe::parse_safetensors(std::make_shared<fe::MappedFile>(path));check(weights.at("w").bytes==4);}
  write(8);rejected=false;try{fe::parse_safetensors(std::make_shared<fe::MappedFile>(path));}catch(...){rejected=true;}check(rejected);fe::fs::remove(path);
  std::cout<<"host assertions: "<<count<<" PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
