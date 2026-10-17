#pragma once
#include "fe/engine.hpp"

namespace fe {
struct ServerConfig {std::string host="127.0.0.1",model="Qwen3",api_key;unsigned short port=8000;bool thinking=true;int reserve_output=512;};
int serve(Engine& engine,Tokenizer& tokenizer,const ServerConfig& config,int64_t vocab_size);
}
