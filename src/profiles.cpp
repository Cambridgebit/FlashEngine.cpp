#include "fe/profiles.hpp"
#include <set>
#include <algorithm>

namespace fe {
bool compatible_contract(const std::string& version){return version=="0.1.0"||version=="0.2.0"||version=="0.3.0"||version==contract_version;}
bool moe_layer(const Json& model,int layer){
  auto kind=model.value("model_type",std::string("qwen3"));if(kind!="qwen3_moe"&&kind!="qwen3_5_moe")return false;
  auto dense=model.at("mlp_only_layers").get<std::vector<int>>();
  return std::find(dense.begin(),dense.end(),layer)==dense.end()&&(layer+1)%model.at("decoder_sparse_step").get<int>()==0;
}
std::map<std::string,std::vector<int64_t>> expected_model_weights(const Json& c,bool draft){
  for(auto key:{"hidden_size","intermediate_size","vocab_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","max_position_embeddings"})require(c.at(key).is_number_integer()&&c.at(key).get<int64_t>()>0,"模型维度非法");
  int64_t h=c.at("hidden_size"),d=c.at("head_dim"),q=c.at("num_attention_heads"),kv=c.at("num_key_value_heads"),i=c.at("intermediate_size"),v=c.at("vocab_size");
  require(d%2==0&&q%kv==0&&h<INT32_MAX&&i<INT32_MAX&&q<INT32_MAX/d,"模型维度溢出/GQA 非法");
  auto family=c.value("model_type",std::string("qwen3"));bool hybrid=family=="qwen3_5"||family=="qwen3_5_moe";
  require(family=="qwen3"||(!draft&&(family=="qwen3_moe"||hybrid)),"不支持的模型 profile");
  if(hybrid){require(c.at("layer_types").size()==c.at("num_hidden_layers").get<size_t>(),"hybrid 层表长度错误");for(auto& type:c.at("layer_types"))require(type=="linear_attention"||type=="full_attention","非法 hybrid 层类型");for(auto name:{"linear_num_key_heads","linear_num_value_heads","linear_key_head_dim","linear_value_head_dim","linear_conv_kernel_dim"})require(c.at(name).get<int64_t>()>0,"非法 GDN 维度");require(c.at("linear_num_value_heads").get<int>()%c.at("linear_num_key_heads").get<int>()==0,"GDN 头数非法");require(c.at("rotary_dim").get<int>()>0&&c.at("rotary_dim").get<int>()<=d&&c.at("rotary_dim").get<int>()%2==0,"非法 partial RoPE");}
  if(family=="qwen3_moe"||family=="qwen3_5_moe"){int e=c.at("num_experts"),k=c.at("num_experts_per_tok");require(e>0&&k>0&&k<=e&&c.at("moe_intermediate_size").get<int>()>0&&c.at("decoder_sparse_step").get<int>()>0,"非法 MoE 配置");std::set<int> seen;for(int layer:c.at("mlp_only_layers").get<std::vector<int>>())require(layer>=0&&layer<c.at("num_hidden_layers").get<int>()&&seen.insert(layer).second,"非法 dense 层列表");}
  std::map<std::string,std::vector<int64_t>> result{{"norm.weight",{h}}};
  if(draft&&c.value("spec_method",std::string("dflash"))=="eagle3"){
    auto dv=c.at("draft_vocab_size").get<int64_t>();require(dv>0&&dv<=v&&c.at("num_hidden_layers")==1,"非法 EAGLE3 维度");
    result["fc.weight"]={h,3*h};result["lm_head.weight"]={dv,h};result["d2t"]={dv};
    result["midlayer.self_attn.q_proj.weight"]={q*d,2*h};result["midlayer.self_attn.k_proj.weight"]={kv*d,2*h};result["midlayer.self_attn.v_proj.weight"]={kv*d,2*h};result["midlayer.self_attn.o_proj.weight"]={h,q*d};
    for(auto name:{"hidden_norm","input_layernorm","post_attention_layernorm"})result[std::string("midlayer.")+name+".weight"]={h};
    result["midlayer.mlp.gate_proj.weight"]={i,h};result["midlayer.mlp.up_proj.weight"]={i,h};result["midlayer.mlp.down_proj.weight"]={h,i};return result;
  }
  if(draft){result["fc.weight"]={h,int64_t(c.at("target_layer_ids").size())*h};result["hidden_norm.weight"]={h};}
  else{result["embed_tokens.weight"]={v,h};if(!c.at("tie_word_embeddings").get<bool>())result["lm_head.weight"]={v,h};}
  for(int l=0;l<c.at("num_hidden_layers").get<int>();++l){auto p="layers."+std::to_string(l)+".";
    result[p+"self_attn.q_proj.weight"]={q*d,h};result[p+"self_attn.k_proj.weight"]={kv*d,h};result[p+"self_attn.v_proj.weight"]={kv*d,h};result[p+"self_attn.o_proj.weight"]={h,q*d};result[p+"self_attn.q_norm.weight"]={d};result[p+"self_attn.k_norm.weight"]={d};result[p+"input_layernorm.weight"]={h};result[p+"post_attention_layernorm.weight"]={h};
    if(hybrid){if(c.at("layer_types").at(l)=="full_attention")result[p+"self_attn.q_proj.weight"]={2*q*d,h};else{
      for(auto name:{"q_proj","k_proj","v_proj","o_proj","q_norm","k_norm"})result.erase(p+"self_attn."+name+".weight");
      int64_t nk=c.at("linear_num_key_heads"),nv=c.at("linear_num_value_heads"),dk=c.at("linear_key_head_dim"),dv=c.at("linear_value_head_dim"),conv=c.at("linear_conv_kernel_dim"),cd=2*nk*dk+nv*dv;auto g=p+"linear_attn.";
      result[g+"in_proj_qkv.weight"]={cd,h};result[g+"in_proj_z.weight"]={nv*dv,h};result[g+"in_proj_b.weight"]={nv,h};result[g+"in_proj_a.weight"]={nv,h};result[g+"conv1d.weight"]={cd,1,conv};result[g+"A_log"]={nv};result[g+"dt_bias"]={nv};result[g+"norm.weight"]={dv};result[g+"out_proj.weight"]={h,nv*dv};
    }}
    if(moe_layer(c,l)){int64_t e=c.at("num_experts"),m=c.at("moe_intermediate_size");result[p+"mlp.gate.weight"]={e,h};auto layouts=c.value("expert_layouts",Json::object());
      auto layout=layouts.value(std::to_string(l),std::string("separate"));require(layout=="fused"||layout=="separate","未知专家权重布局");
      if(layout=="fused"){result[p+"mlp.experts.gate_up_proj.weight"]={e,2*m,h};result[p+"mlp.experts.down_proj.weight"]={e,h,m};}
      else for(int expert=0;expert<e;++expert){auto x=p+"mlp.experts."+std::to_string(expert)+".";result[x+"gate_proj.weight"]={m,h};result[x+"up_proj.weight"]={m,h};result[x+"down_proj.weight"]={h,m};}
      if(family=="qwen3_5_moe"){int64_t width=c.at("shared_expert_intermediate_size");require(width>0,"非法共享专家宽度");result[p+"mlp.shared_expert.gate_proj.weight"]={width,h};result[p+"mlp.shared_expert.up_proj.weight"]={width,h};result[p+"mlp.shared_expert.down_proj.weight"]={h,width};result[p+"mlp.shared_expert_gate.weight"]={1,h};}
    }else{result[p+"mlp.gate_proj.weight"]={i,h};result[p+"mlp.up_proj.weight"]={i,h};result[p+"mlp.down_proj.weight"]={h,i};}
  }return result;
}
}
