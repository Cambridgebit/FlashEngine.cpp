#include "ascend_qwen3.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

int main(){try{
  using namespace fe::ascend;
  Runtime runtime;
  {
    auto cache=qwen_rope_cache(256,128,1000000.f,runtime.stream);auto cs=download(cache.first,runtime.stream),sn=download(cache.second,runtime.stream);auto cos=reinterpret_cast<const uint16_t*>(cs.data.data()),sin=reinterpret_cast<const uint16_t*>(sn.data.data());
    // 来自相同 CANN 环境下 Python FP32 RoPE -> FP16 的实际值，覆盖主机 libm 会舍入不同的点。
    fe::require(sin[23*128+6]==9157&&cos[27*128+3]==8229&&sin[53*128+8]==34480&&sin[60*128]==46304,"NPU RoPE 常量与 Python 的舍入不一致");
    for(int row=0;row<256;++row)for(int j=0;j<64;++j)fe::require(cos[row*128+j]==cos[row*128+j+64]&&sin[row*128+j]==sin[row*128+j+64],"RoPE half 模式常量布局错误");
  }
  std::cout<<"NPU RoPE 常量：Python 舍入边界及 half 布局 PASS\n";
  for(bool graph:{false,true}){
    Program p;auto storage=p.out({1,6*1024}),key=p.out({1,8,128}),query=p.out({1,32,128}),slots=p.cast(p.indices({1}),ACL_INT32),table=p.view(p.cast(p.indices({0,1}),ACL_INT32),{1,2}),length=p.cast(p.indices({2}),ACL_INT32),output=p.out({1,32,128});zero(key);zero(query);auto value=p.view(p.narrow(storage,1,1024,1024),{1,8,128});fe::require(atb_tensor(value).dataSize==2048,"ND 子视图暴露了底层存储容量");auto kc=physical_nz({3,64,128,16}),vc=physical_nz({3,64,128,16});cache_kv(p,key,value,kc,vc,slots);atb::infer::PagedAttentionParam param;param.headNum=32;param.kvHeadNum=8;param.qkScale=1.f/std::sqrt(128.f);p.atb(param,{query,kc,vc,table,length},{output});p.finish();std::vector<uint16_t> values(6*1024,fp16(3));upload(storage,values.data(),values.size()*2);if(graph)p.capture(runtime.stream);for(int pass=0;pass<2;++pass){if(pass){std::fill(values.begin(),values.end(),fp16(4));upload(storage,values.data(),values.size()*2);}if(graph)p.run(runtime.stream);else p.eager(runtime.stream);auto result=download(output,runtime.stream);auto actual=reinterpret_cast<const uint16_t*>(result.data.data());for(int j=0;j<32*128;++j)fe::require(std::abs(int(actual[j])-int(fp16(float(pass?4:3)/2)))<=3,"ND 单行子视图到 NZ KV 的 eager/Graph 执行错误");}
  }
  std::cout<<"ND 单行投影切片：有效字节数、偏移、Paged decode、eager/Graph PASS\n";
  for(bool dummy:{false,true}){
    fe::Config c;c.max_model_len=256;c.num_kvcache_blocks=2;c.kvcache_block_size=128;Program p;QwenMeta meta(p,c,{32},true);auto q=p.out({32,32,128}),k=p.out({32,8,128}),v=p.out({32,8,128}),cos=p.out({32,128}),sin=p.out({32,128}),length=p.cast(p.indices({32}),ACL_INT32);std::vector<uint16_t> qv(32*32*128,fp16(1)),vv(32*8*128),ones(32*128,fp16(1));upload(q,qv.data(),qv.size()*2);zero(k);zero(cos);upload(sin,ones.data(),ones.size()*2);for(int row=0;row<32;++row)for(int j=0;j<8*128;++j)vv[row*8*128+j]=fp16(row+1);upload(v,vv.data(),vv.size()*2);auto pair=rotary_pair(p,q,k,cos,sin,length);auto kc=physical_nz({3,64,128,16}),vc=physical_nz({3,64,128,16});cache_kv(p,pair.second,v,kc,vc,meta.slots);auto output=p.out({32,32,128});atb::infer::SelfAttentionParam param;param.headNum=32;param.kvHeadNum=8;param.qkScale=1.f/std::sqrt(128.f);param.calcType=atb::infer::SelfAttentionParam::PA_ENCODER;param.maskType=atb::infer::SelfAttentionParam::MASK_TYPE_NORM;p.atb(param,{pair.first,pair.second,v,meta.mask},{output},meta.host_qlens);p.finish();fe::Seq seq;if(!dummy){seq=std::make_shared<fe::Sequence>(1,std::vector<int64_t>(32,0),1);seq->blocks={0,1};}meta.prepare({seq},c,{0},{32});p.eager(runtime.stream);auto result=download(output,runtime.stream);auto values=reinterpret_cast<const uint16_t*>(result.data.data());for(int row=0;row<32;++row)for(int j=0;j<32*128;++j)fe::require(std::abs(int(values[row*32*128+j])-int(fp16(float(row+2)/2)))<=3,"8B 头形状 RoPE/KV/Flash 管线错误");
  }
  std::cout<<"8B 头形状：RoPE 输出到 NZ KV / Flash，真实页及全 dummy 页 PASS\n";
  {
    Program convert;auto input=tensor({16,128}),weight=tensor({64,128});std::vector<uint16_t> x(16*128),w(64*128,0),expected(16*64);for(int row=0;row<16;++row)for(int j=0;j<128;++j)x[row*128+j]=fp16(float((j%17)-8)/8+float(row)/4);for(int j=0;j<64;++j){int column=(j*2+3)%128;w[j*128+column]=fp16(1);for(int row=0;row<16;++row)expected[row*64+j]=x[row*128+column];}upload(input,x.data(),x.size()*2);upload(weight,w.data(),w.size()*2);auto nz=convert.nz(weight);convert.finish();convert.eager(runtime.stream);check(aclrtSynchronizeStream(runtime.stream),"非方阵 NZ 转换");Program p;auto output=p.linear(input,nz);p.finish();p.eager(runtime.stream);auto result=download(output,runtime.stream);fe::require(!std::memcmp(result.data.data(),expected.data(),expected.size()*2),"非方阵 NZ 转置或元素布局错误");
  }
  std::cout<<"ATB NZ 非方阵线性层布局与转置 PASS\n";
  for(std::string dtype:{"BF16","F32"}){
    size_t count=1<<20;auto file=fe::fs::temp_directory_path()/("fe-ascend-weights-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");std::vector<uint16_t> bf(count);std::vector<float> floats(count);std::vector<uint16_t> expected(count);for(size_t i=0;i<count;++i){bf[i]=uint16_t(i*7919);floats[i]=std::bit_cast<float>(uint32_t(bf[i])<<16);expected[i]=fp16(floats[i]);}
    {std::ofstream out(file,std::ios::binary);out.write(dtype=="BF16"?reinterpret_cast<const char*>(bf.data()):reinterpret_cast<const char*>(floats.data()),count*(dtype=="BF16"?2:4));}
    std::map<std::string,fe::WeightView> weights{{"weight",{std::make_shared<fe::MappedFile>(file),0,count*(dtype=="BF16"?2:4),dtype,{1024,1024}}}};fe::fs::remove(file);auto result=download(load_weight(weights,"weight"),runtime.stream);fe::require(!std::memcmp(result.data.data(),expected.data(),count*2),"权重转换与逐元素 IEEE-754 舍入不同");result=download(load_weight(weights,"weight",true),runtime.stream);fe::require(!std::memcmp(result.data.data(),floats.data(),count*4),"FP32 权重转换不同");
  }
  std::cout<<"BF16/F32 大权重转换：完整位模式、FP16/FP32 输出 PASS\n";
  for(bool graph:{false,true}){
    Program p;auto x=p.out({16,1,128}),cos=p.out({16,128}),sin=p.out({16,128}),length=p.cast(p.indices({16}),ACL_INT32);std::vector<uint16_t> values(2048,fp16(2)),cs(2048,fp16(0)),sn(2048,fp16(1));upload(x,values.data(),4096);upload(cos,cs.data(),4096);upload(sin,sn.data(),4096);auto pair=rotary_pair(p,x,x,cos,sin,length);p.finish();if(graph)p.capture(runtime.stream);
    for(int pass=0;pass<2;++pass){if(pass){std::fill(values.begin(),values.end(),fp16(3));upload(x,values.data(),4096);}if(graph)p.run(runtime.stream);else p.eager(runtime.stream);for(auto t:{pair.first,pair.second}){auto result=download(t,runtime.stream);auto actual=reinterpret_cast<const uint16_t*>(result.data.data());for(int row=0;row<16;++row)for(int j=0;j<128;++j)fe::require(actual[row*128+j]==fp16(float(pass?3:2)*(j<64?-1:1)),"RoPE Q/K 别名导致重复旋转");}auto source=download(x,runtime.stream);fe::require(!std::memcmp(source.data.data(),values.data(),4096),"RoPE 修改了共享上下文输入");}
  }
  std::cout<<"RoPE 共享 Q/K 输入：非零角、输入不变、eager/Graph 动态值 PASS\n";
  {
    Program p;auto q=p.out({16,256}),k=p.out({16,128}),cos=p.out({16,128}),sin=p.out({16,128}),length=p.cast(p.indices({16}),ACL_INT32);std::vector<uint16_t> qv(4096,fp16(2)),kv(2048,fp16(3)),cv(2048,fp16(1));upload(q,qv.data(),8192);upload(k,kv.data(),4096);upload(cos,cv.data(),4096);zero(sin);atb::infer::RopeParam param;param.rotaryCoeff=2;p.atb(param,{q,k,cos,sin,length},{q,k});p.finish();p.eager(runtime.stream);auto actual=download(q,runtime.stream);fe::require(!std::memcmp(actual.data.data(),qv.data(),8192),"ATB RoPE Q 结果错误");actual=download(k,runtime.stream);fe::require(!std::memcmp(actual.data.data(),kv.data(),4096),"ATB RoPE K 结果错误");
  }
  std::cout<<"ATB RoPE 半分旋转、不同 Q/K 头数 PASS\n";
  {
    fe::Config c;c.max_model_len=256;c.num_kvcache_blocks=2;c.kvcache_block_size=128;Program p;QwenMeta meta(p,c,{32},true);auto q=p.out({32,2,128}),k=p.out({32,1,128}),v=p.out({32,1,128}),out=p.out({32,2,128});zero(q);zero(k);std::vector<uint16_t> values(32*128);for(int row=0;row<32;++row)for(int j=0;j<128;++j)values[row*128+j]=fp16(row+1);upload(v,values.data(),values.size()*2);meta.prepare({{}},c,{0},{32});atb::infer::SelfAttentionParam param;param.headNum=2;param.kvHeadNum=1;param.qkScale=1.f/std::sqrt(128.f);param.calcType=atb::infer::SelfAttentionParam::PA_ENCODER;param.maskType=atb::infer::SelfAttentionParam::MASK_TYPE_NORM;p.atb(param,{q,k,v,meta.mask},{out},meta.host_qlens);p.finish();p.eager(runtime.stream);auto result=download(out,runtime.stream);auto actual=reinterpret_cast<const uint16_t*>(result.data.data());for(int row=0;row<32;++row)for(int j=0;j<256;++j)fe::require(std::abs(int(actual[row*256+j])-int(fp16(float(row+2)/2)))<=3,"ATB Flash Attention 因果 mask 错误");
  }
  std::cout<<"ATB Flash Attention 批量 prefill 因果 mask PASS\n";
  {
    // 非等长序列必须各自从零开始应用因果掩码，不能互相读取前一条的 V。
    fe::Config c;c.max_model_len=256;c.num_kvcache_blocks=2;c.kvcache_block_size=128;Program p;QwenMeta meta(p,c,{3,13},true);auto q=p.out({16,2,128}),k=p.out({16,1,128}),v=p.out({16,1,128}),out=p.out({16,2,128});zero(q);zero(k);std::vector<uint16_t> values(16*128);
    for(int row=0;row<16;++row)for(int j=0;j<128;++j)values[row*128+j]=fp16(row<3?row+1:100+row-3);upload(v,values.data(),values.size()*2);meta.prepare({{},{}},c,{0,0},{3,13});atb::infer::SelfAttentionParam param;param.headNum=2;param.kvHeadNum=1;param.qkScale=1.f/std::sqrt(128.f);param.calcType=atb::infer::SelfAttentionParam::PA_ENCODER;param.maskType=atb::infer::SelfAttentionParam::MASK_TYPE_NORM;p.atb(param,{q,k,v,meta.mask},{out},meta.host_qlens);p.finish();p.eager(runtime.stream);auto result=download(out,runtime.stream);auto actual=reinterpret_cast<const uint16_t*>(result.data.data());
    for(int row=0;row<16;++row)for(int j=0;j<256;++j)fe::require(std::abs(int(actual[row*256+j])-int(fp16(row<3?float(row+2)/2:100.f+float(row-3)/2)))<=3,"Flash Attention 非等长序列的因果掩码串扰");
  }
  std::cout<<"ATB Flash Attention 非等长双序列隔离 PASS\n";
  for(bool graph:{false,true}){
    Program p;auto proposed=p.out({3,3},ACL_INT64),verified=p.out({3,4},ACL_INT64),slots=p.out({12},ACL_INT32);std::vector<int64_t> d{5,6,7,1,2,3,11,12,13},v{9,6,7,8,1,9,3,4,11,12,13,14};std::vector<int32_t> ids(12);std::iota(ids.begin(),ids.end(),0);upload(proposed,d.data(),72);upload(verified,v.data(),96);upload(slots,ids.data(),48);
    auto commit=greedy_device_commit(p,proposed,verified,slots,256);auto output=p.cat({commit.first,p.view(p.cast(commit.second,ACL_INT64),{3,4})},1);p.finish();if(graph)p.capture(runtime.stream);
    for(int pass=0;pass<2;++pass){std::vector<int64_t> counts=pass?std::vector<int64_t>{4,2,1}:std::vector<int64_t>{1,2,4};if(pass){v[0]=5;v[8]=99;upload(verified,v.data(),96);}if(graph)p.run(runtime.stream);else p.eager(runtime.stream);auto result=read_ids(output,runtime.stream);for(int i=0;i<3;++i){fe::require(result[i*5]==counts[i],"设备接受长度错误");for(int j=0;j<4;++j)fe::require(result[i*5+j+1]==(j<counts[i]?i*4+j:256+j),"设备接受槽位掩码错误");}}
  }
  std::cout<<"设备投机提交：零/部分/全部接受、批量 eager/Graph、动态 token PASS\n";
  for(bool graph:{false,true}){
    fe::Config c;c.max_model_len=256;c.num_kvcache_blocks=2;c.kvcache_block_size=128;
    Program p;QwenMeta meta(p,c,{16});auto key=p.out({16,1,128}),value=p.out({16,1,128}),query=p.out({16,2,128}),output=p.out({16,2,128});zero(key);zero(query);auto kc=physical_nz({3,8,128,16}),vc=physical_nz({3,8,128,16});
    cache_kv(p,key,value,kc,vc,meta.slots);atb::infer::PagedAttentionParam param;param.headNum=2;param.kvHeadNum=1;param.qkScale=1.f/std::sqrt(128.f);param.maskType=atb::infer::PagedAttentionParam::MASK_TYPE_SPEC;param.calcType=atb::infer::PagedAttentionParam::CALC_TYPE_SPEC;p.atb(param,{query,kc,vc,meta.table,meta.lens,meta.mask},{output},meta.host_qlens);p.finish();
    auto seq=std::make_shared<fe::Sequence>(1,std::vector<int64_t>(32,0),32);seq->blocks={0,1};std::vector<uint16_t> values(16*128,fp16(1));upload(value,values.data(),values.size()*2);meta.prepare({seq},c,{0},{16});if(graph){p.capture(runtime.stream);check(aclrtSynchronizeStream(runtime.stream),"ATB Graph 捕获同步");zero(kc);zero(vc);}
    for(int pass=0;pass<2;++pass){int start=pass*16;for(int row=0;row<16;++row)for(int j=0;j<128;++j)values[row*128+j]=fp16(float(start+row+1));upload(value,values.data(),values.size()*2);meta.prepare({seq},c,{start},{16});if(graph)p.run(runtime.stream);else p.eager(runtime.stream);auto result=download(output,runtime.stream);auto actual=reinterpret_cast<const uint16_t*>(result.data.data());for(int row=0;row<16;++row)for(int j=0;j<256;++j)fe::require(std::abs(int(actual[row*256+j])-int(fp16(float(start+row+2)/2)))<=3,"ATB SplitFuse 动态上下文或因果 mask 错误");}
  }
  std::cout<<"ATB NZ KV + SplitFuse：eager/Graph 动态长度和因果 mask PASS\n";
  {
    auto x=tensor({16,128}),gamma=tensor({128}),weight=tensor({128,128});std::vector<uint16_t> data(16*128,fp16(1)),ones(128,fp16(1)),identity(128*128,0);for(int i=0;i<128;++i)identity[i*128+i]=fp16(1);
    upload(x,data.data(),data.size()*2);upload(gamma,ones.data(),ones.size()*2);upload(weight,identity.data(),identity.size()*2);
    Program convert;auto nz=convert.nz(weight);convert.finish();convert.eager(runtime.stream);check(aclrtSynchronizeStream(runtime.stream),"NZ 转换同步");
    Program p;auto projected=p.linear(x,nz),norm=p.fast_norm(projected,gamma,1e-6);auto added=p.add_norm(x,x,gamma,1e-6);auto glu=p.swiglu(p.cat({x,x},1));p.finish();p.eager(runtime.stream);
    auto result=download(projected,runtime.stream);fe::require(result.data.size()==data.size()*2&&!std::memcmp(result.data.data(),data.data(),result.data.size()),"ATB NZ 矩阵乘精度失败");
    auto check_close=[&](const T& t,float target){auto h=download(t,runtime.stream);const uint16_t* v=reinterpret_cast<const uint16_t*>(h.data.data());for(size_t i=0;i<h.data.size()/2;++i)fe::require(std::abs(int(v[i])-int(fp16(target)))<=2,"融合算子数值失败");};
    check_close(norm,1);check_close(added.first,1);check_close(added.second,2);check_close(glu,1.f/(1.f+std::exp(-1.f)));
    std::cout<<"ATB NZ Linear / FP16 RMSNorm / AddRMSNorm / SwiGLU PASS\n";
  }
  for(bool graph:{false,true})for(int64_t rows:{1,2}){
    auto cache=tensor({6,4}),ids=tensor({rows},ACL_INT64),values=tensor({rows,4});zero(cache);
    std::vector<int64_t> slots{1,3};slots.resize(rows);
    std::vector<uint16_t> input(rows*4),expected(24,0);
    for(size_t i=0;i<input.size();++i)input[i]=fp16(float(i+1));
    upload(ids,slots.data(),slots.size()*8);upload(values,input.data(),input.size()*2);
    Program program;program.scatter(cache,ids,values);program.finish();
    if(graph){program.capture(runtime.stream);check(aclrtSynchronizeStream(runtime.stream),"KV Graph 捕获同步");zero(cache);}
    for(int pass=0;pass<2;++pass){
      if(pass){for(auto& slot:slots)++slot;for(auto& value:input)value=fp16(16.f);upload(ids,slots.data(),slots.size()*8);upload(values,input.data(),input.size()*2);}
      if(graph)program.run(runtime.stream);else program.eager(runtime.stream);
      check(aclrtSynchronizeStream(runtime.stream),"动态槽位执行");
      for(int64_t row=0;row<rows;++row)for(int j=0;j<4;++j)expected[slots[row]*4+j]=input[row*4+j];
      std::vector<uint16_t> output(24);check(aclrtMemcpy(output.data(),48,cache->data(),48,ACL_MEMCPY_DEVICE_TO_HOST),"动态槽位读回");
      fe::require(output==expected,"KV 更新覆盖了未选中的槽位，或 executor 固化了旧索引/数值");
    }
  }
  std::cout<<"Ascend KV 动态槽位：eager/Graph、单行/多行、重复执行、未选中行保持 PASS\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
