#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fe {
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;
inline constexpr const char* contract_version = "0.4.0";
struct InputError : std::runtime_error { using std::runtime_error::runtime_error; };
void require(bool condition, const std::string& message);
Json parse_json(const std::string& text);
Json read_json(const fs::path& path);
std::string read_text(const fs::path& path);
std::string sha256(const fs::path& path);
int64_t parse_integer(const std::string& text);
std::vector<int64_t> graph_buckets(int64_t maximum);

struct Config {
  int64_t max_model_len=2048, max_num_seqs=8, max_num_batched_tokens=2048;
  int64_t max_num_partial_prefills=1, kvcache_block_size=256, num_kvcache_blocks=64;
  int64_t eos=-1;
  int64_t spec_num_tokens=0;
  bool enable_chunked_prefill=true, enforce_eager=true, production=false;
  bool trace_logits=false, enable_prefix_cache=true;
  int64_t recurrent_prefix_cache_bytes=256LL*1024*1024;
  std::string dtype="bfloat16", backend="cuda";
  void validate() const;
  Json json() const;
};
// 公共类型仅描述内存和布局；不携带 ATen/CUDA/ACL 对象。
enum class ScalarType { fp16, bf16, fp32, i32, i64 };
struct TensorDesc { ScalarType dtype; std::vector<int64_t> shape, strides; };
struct TensorView { void* data=nullptr; TensorDesc desc; };
struct DeviceBuffer { virtual ~DeviceBuffer()=default; virtual TensorView view()=0; };
struct BackendCapabilities {
  bool graph=false, speculative=false, prefix_cache=false;
  std::string device, kv_layout;
};
struct ExecutionPlan { virtual ~ExecutionPlan()=default; };
}
