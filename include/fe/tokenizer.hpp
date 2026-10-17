#pragma once
#include "fe/common.hpp"
#include <memory>

namespace fe {
class Tokenizer {
  struct Impl;std::shared_ptr<Impl> impl_;
public:
  explicit Tokenizer(const fs::path& bundle_root);
  std::vector<int64_t> encode(const std::string& text)const;
  std::string decode(const std::vector<int64_t>& ids,bool skip_special=true)const;
  std::string render(const Json& messages,bool thinking=true)const;
  void verify_golden(const Json& golden)const;
};
}
