#include "fe/common.hpp"
#include <openssl/evp.h>
#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>

namespace fe {
int64_t parse_integer(const std::string& text) {
  int64_t result=0;auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),result);
  require(!text.empty()&&error==std::errc{}&&end==text.data()+text.size(),"非法整数："+text);
  return result;
}
void require(bool value,const std::string& message) { if(!value) throw std::runtime_error(message); }
Json parse_json(const std::string& text) {
  std::vector<std::set<std::string>> keys;
  return Json::parse(text,[&](int,Json::parse_event_t event,Json& value) {
    if(event==Json::parse_event_t::object_start) keys.emplace_back();
    if(event==Json::parse_event_t::key) require(keys.back().insert(value.get<std::string>()).second,"重复 JSON 键");
    if(event==Json::parse_event_t::object_end) keys.pop_back();
    return true;
  });
}
std::string read_text(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary);
  require(stream.good(),"无法打开文件："+path.string());
  return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
Json read_json(const fs::path& path) {return parse_json(read_text(path));}
std::string sha256(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary);
  require(stream.good(),"哈希文件无法打开："+path.string());
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  require(ctx && EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA256 初始化失败");
  std::array<char,1<<20> buffer;
  while(stream) {stream.read(buffer.data(),buffer.size());require(EVP_DigestUpdate(ctx.get(),buffer.data(),stream.gcount())==1,"SHA256 更新失败");}
  require(stream.eof(),"哈希读取失败");
  unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int size=0;
  require(EVP_DigestFinal_ex(ctx.get(),digest,&size)==1,"SHA256 完成失败");
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(unsigned i=0;i<size;++i)out<<std::setw(2)<<unsigned(digest[i]);return out.str();
}
std::vector<int64_t> graph_buckets(int64_t maximum) {
  require(maximum>0,"图桶上限必须为正");std::vector<int64_t> result;
  for(int n:{1,2,4,8})if(n<=maximum)result.push_back(n);
  for(int64_t n=16;n<=maximum;n+=16)result.push_back(n);
  if(result.empty()||result.back()!=maximum)result.push_back(maximum);
  return result;
}
void Config::validate() const {
  require(max_model_len>=2 && max_num_seqs>0 && max_num_batched_tokens>0,"非法上下文/批次配置");
  require(max_num_partial_prefills>0 && max_num_partial_prefills<=max_num_seqs,"非法 partial prefill 数量");
  require(enable_chunked_prefill||max_num_batched_tokens>=max_model_len,"关闭分块时 token 预算必须覆盖上下文");
  require(kvcache_block_size>=16 && !(kvcache_block_size&(kvcache_block_size-1)),"KV block 必须是至少 16 的 2 次幂");
  require(num_kvcache_blocks>0,"KV 页数必须为正");
  require(num_kvcache_blocks<INT32_MAX/kvcache_block_size,"KV 物理槽超出 int32 范围");
  require((max_model_len-1)/kvcache_block_size+1<=num_kvcache_blocks,"KV 池必须至少容纳一条完整上下文，避免不可恢复的抢占循环");
  require(max_model_len<=INT32_MAX&&max_num_seqs<=1024&&max_num_batched_tokens<=INT32_MAX,"配置超出原生索引范围");
  require(spec_num_tokens>=0&&spec_num_tokens<max_model_len,"非法投机长度");
  require(dtype=="float16"||dtype=="bfloat16","不支持的 dtype");
  require(backend=="cuda"||backend=="corex"||backend=="ascend310p","未知后端");
  require(!(production&&enforce_eager),"production 要求 Graph");
}
Json Config::json() const {return {{"max_model_len",max_model_len},{"max_num_seqs",max_num_seqs},{"max_num_batched_tokens",max_num_batched_tokens},{"max_num_partial_prefills",max_num_partial_prefills},{"kvcache_block_size",kvcache_block_size},{"num_kvcache_blocks",num_kvcache_blocks},{"enable_chunked_prefill",enable_chunked_prefill},{"enable_prefix_cache",false},{"enforce_eager",enforce_eager},{"production",production},{"backend",backend},{"dtype",dtype},{"tensor_parallel_size",1},{"spec_num_tokens",spec_num_tokens},{"eos_token_id",eos},{"trace_logits",trace_logits}};}
}
