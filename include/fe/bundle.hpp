#pragma once
#include "fe/common.hpp"
#include <map>
#include <memory>

namespace fe {
class MappedFile {
  void* data_=nullptr; size_t size_=0;
public:
  explicit MappedFile(const fs::path& path);
  ~MappedFile();
  MappedFile(const MappedFile&)=delete;
  MappedFile& operator=(const MappedFile&)=delete;
  const char* data()const{return static_cast<const char*>(data_);}
  size_t size()const{return size_;}
};
struct WeightView {
  std::shared_ptr<MappedFile> file;
  size_t offset=0, bytes=0;
  std::string dtype;
  std::vector<int64_t> shape;
  const void* data()const{return file->data()+offset;}
};
std::map<std::string,WeightView> parse_safetensors(std::shared_ptr<MappedFile> file);
class Bundle {
public:
  fs::path root;
  Json manifest, config, tokenizer_runtime;
  std::map<std::string,WeightView> target_weights, draft_weights;
  explicit Bundle(const fs::path& directory,bool verify_hashes=true);
  fs::path path(const std::string& relative)const;
  void validate_weights();
};
}
