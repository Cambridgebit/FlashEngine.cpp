#pragma once
#include "fe/common.hpp"
#include <map>

namespace fe {
bool compatible_contract(const std::string& version);
bool moe_layer(const Json& model,int layer);
std::map<std::string,std::vector<int64_t>> expected_model_weights(const Json& model,bool draft);
}
