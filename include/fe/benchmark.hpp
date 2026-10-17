#pragma once
#include "fe/model.hpp"

namespace fe {
// 仅基准入口；预先冻结的 token 输入避免把两种 tokenizer/模板开销混入引擎时间。
int run_benchmark_cli(const Bundle& bundle,int argc,char** argv,const std::string& backend,const DeviceFactory& factory);
}
