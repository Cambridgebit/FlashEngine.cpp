#pragma once
#include "fe/model.hpp"
#include <ATen/ATen.h>
namespace fe {
// 设备内部参考接口，不向调度器暴露 ATen。
class BlockDraft {
public:
  virtual ~BlockDraft()=default;
  virtual void project(const at::Tensor& aux,const Seq& sequence,size_t position)=0;
  virtual std::vector<int64_t> propose(const Seq& sequence)=0;
  virtual Json snapshot()const=0;
};
std::unique_ptr<BlockDraft> create_block_draft(const Bundle&,const Config&,at::Tensor embed,at::Tensor head);
}
