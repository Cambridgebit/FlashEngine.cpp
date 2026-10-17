#include "fe/profiles.hpp"
#include <set>
#include <algorithm>

namespace fe {
bool compatible_contract(const std::string& version){return version=="0.1.0"||version==contract_version;}
bool moe_layer(const Json& model,int layer){
  if(model.value("model_type",std::string("qwen3"))!="qwen3_moe")return false;
  auto dense=model.at("mlp_only_layers").get<std::vector<int>>();
  return std::find(dense.begin(),dense.end(),layer)==dense.end()&&(layer+1)%model.at("decoder_sparse_step").get<int>()==0;
}
std::map<std::string,std::vector<int64_t>> expected_model_weights(const Json& c,bool draft){
  for(auto key:{"hidden_size","intermediate_size","vocab_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","max_position_embeddings"})require(c.at(key).is_number_integer()&&c.at(key).get<int64_t>()>0,"模型维度非法");
  int64_t h=c.at("hidden_size"),d=c.at("head_dim"),q=c.at("num_attention_heads"),kv=c.at("num_key_value_heads"),i=c.at("intermediate_size"),v=c.at("vocab_size");
  require(d%2==0&&q%kv==0&&h<INT32_MAX&&i<INT32_MAX&&q<INT32_MAX/d,"模型维度溢出/GQA 非法");
  auto family=c.value("model_type",std::string("qwen3"));require(family=="qwen3"||(!draft&&family=="qwen3_moe"),"不支持的模型 profile");
  if(family=="qwen3_moe"){int e=c.at("num_experts"),k=c.at("num_experts_per_tok");require(e>0&&k>0&&k<=e&&c.at("moe_intermediate_size").get<int>()>0&&c.at("decoder_sparse_step").get<int>()>0,"非法 MoE 配置");std::set<int> seen;for(int layer:c.at("mlp_only_layers").get<std::vector<int>>())require(layer>=0&&layer<c.at("num_hidden_layers").get<int>()&&seen.insert(layer).second,"非法 dense 层列表");}
  std::map<std::string,std::vector<int64_t>> result{{"norm.weight",{h}}};
  if(draft){result["fc.weight"]={h,int64_t(c.at("target_layer_ids").size())*h};result["hidden_norm.weight"]={h};}
  else{result["embed_tokens.weight"]={v,h};if(!c.at("tie_word_embeddings").get<bool>())result["lm_head.weight"]={v,h};}
  for(int l=0;l<c.at("num_hidden_layers").get<int>();++l){auto p="layers."+std::to_string(l)+".";
    result[p+"self_attn.q_proj.weight"]={q*d,h};result[p+"self_attn.k_proj.weight"]={kv*d,h};result[p+"self_attn.v_proj.weight"]={kv*d,h};result[p+"self_attn.o_proj.weight"]={h,q*d};result[p+"self_attn.q_norm.weight"]={d};result[p+"self_attn.k_norm.weight"]={d};result[p+"input_layernorm.weight"]={h};result[p+"post_attention_layernorm.weight"]={h};
    if(moe_layer(c,l)){int64_t e=c.at("num_experts"),m=c.at("moe_intermediate_size");result[p+"mlp.gate.weight"]={e,h};auto layouts=c.value("expert_layouts",Json::object());
      auto layout=layouts.value(std::to_string(l),std::string("separate"));require(layout=="fused"||layout=="separate","未知专家权重布局");
      if(layout=="fused"){result[p+"mlp.experts.gate_up_proj.weight"]={e,2*m,h};result[p+"mlp.experts.down_proj.weight"]={e,h,m};}
      else for(int expert=0;expert<e;++expert){auto x=p+"mlp.experts."+std::to_string(expert)+".";result[x+"gate_proj.weight"]={m,h};result[x+"up_proj.weight"]={m,h};result[x+"down_proj.weight"]={h,m};}
    }else{result[p+"mlp.gate_proj.weight"]={i,h};result[p+"mlp.up_proj.weight"]={i,h};result[p+"mlp.down_proj.weight"]={h,i};}
  }return result;
}
}
