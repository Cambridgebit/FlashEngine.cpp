#pragma once
#include "fe/bundle.hpp"
#include "fe/scheduler.hpp"
#include "fe/tokenizer.hpp"

namespace fe {
struct ModelOutput {std::vector<std::vector<int64_t>> tokens;Json rounds=Json::array(),samples=Json::array();};
class DeviceModel {
public:
  virtual ~DeviceModel()=default;
  virtual ModelOutput run(const Batch& batch)=0;
  virtual void drop(uint64_t id)=0;
  virtual Json snapshot()const=0;
};
std::unique_ptr<DeviceModel> create_cuda_model(const Bundle& bundle,const Config& config);
using DeviceFactory=std::function<std::unique_ptr<DeviceModel>(const Bundle&,const Config&)>;
int run_device_cli(const std::string& command,Bundle& bundle,Tokenizer& tokenizer,int argc,char** argv,const std::string& backend,const DeviceFactory& factory,const std::string& default_dtype="bfloat16");
std::unique_ptr<DeviceModel> create_ascend_model(const Bundle& bundle,const Config& config);
}
