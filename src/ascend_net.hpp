#pragma once
// 全 C++ ACLNN 参考网络；没有 TorchNPU、Python 或设备 JIT 依赖。
#include "ascend_ops.hpp"
#include "fe/bundle.hpp"
#include "fe/profiles.hpp"
#include <cmath>
#include <list>
#include <map>

namespace fe::ascend {
struct Runtime {
  aclrtContext context=nullptr;aclrtStream stream=nullptr;
  Runtime(){check(aclInit(nullptr),"aclInit");try{check(aclrtSetDevice(0),"aclrtSetDevice");check(aclrtCreateContext(&context,0),"aclrtCreateContext");check(aclrtSetCurrentContext(context),"初始化 NPU context");check(aclrtCreateStream(&stream),"aclrtCreateStream");}catch(...){if(context)aclrtDestroyContext(context);aclrtResetDevice(0);aclFinalize();throw;}}
  ~Runtime(){if(context)aclrtSetCurrentContext(context);if(stream){aclrtSynchronizeStream(stream);aclrtDestroyStream(stream);}if(context)aclrtDestroyContext(context);aclrtResetDevice(0);aclFinalize();}
  void enter(){check(aclrtSetCurrentContext(context),"切换 NPU context");}
};
inline float host_float(const WeightView& w,size_t i){
  require(i<size_t(elements(w.shape)),"权重读取越界");auto p=static_cast<const char*>(w.data());
  if(w.dtype=="F32"){float v;std::memcpy(&v,p+i*4,4);return v;}
  if(w.dtype=="I8")return static_cast<const int8_t*>(w.data())[i];
  uint16_t bits;require(w.dtype=="F16"||w.dtype=="BF16","不支持的浮点源 dtype");std::memcpy(&bits,p+i*2,2);
  if(w.dtype=="BF16")return std::bit_cast<float>(uint32_t(bits)<<16);
  int e=(bits>>10)&31;float n=std::ldexp(float(bits&1023),-24);if(e==31)n=(bits&1023)?NAN:INFINITY;else if(e)n=std::ldexp(float(1024+(bits&1023)),e-25);return bits&0x8000?-n:n;
}
inline T load_weight(const std::map<std::string,WeightView>& source,const std::string& name,bool fp32=false){
  auto& w=source.at(name);if(w.dtype=="I64"){auto out=tensor(w.shape,ACL_INT64);upload(out,w.data(),w.bytes);return out;}
  auto out=tensor(w.shape,fp32?ACL_FLOAT:ACL_FLOAT16);size_t count=elements(w.shape),chunk=std::min(count,size_t(1<<18));std::vector<float> converted(chunk);std::vector<uint16_t> half(chunk);
  std::vector<float> scale,zero;if(w.dtype=="I8"){
    require(name.ends_with(".weight")&&w.shape.size()==2,"Quark 只支持逐输出通道 2D int8 权重");auto p=name.substr(0,name.size()-7);auto& s=source.at(p+".weight_scale");require(elements(s.shape)==w.shape[0],"Quark scale 形状错误");
    for(int64_t r=0;r<w.shape[0];++r){float v=host_float(s,r);require(std::isfinite(v)&&v>0,"Quark scale 必须有限且为正");scale.push_back(v);zero.push_back(source.contains(p+".weight_zero_point")?host_float(source.at(p+".weight_zero_point"),r):0);require(std::isfinite(zero.back()),"Quark zero 非有限");}
  }
  for(size_t start=0;start<count;start+=chunk){size_t n=std::min(chunk,count-start);for(size_t i=0;i<n;++i){float v=host_float(w,start+i);if(!scale.empty()){size_t row=(start+i)/w.shape[1];v=(v-zero[row])*scale[row];}converted[i]=v;half[i]=fp16(v);}size_t item=fp32?4:2;check(aclrtMemcpy(static_cast<char*>(out->data())+start*item,(count-start)*item,fp32?static_cast<void*>(converted.data()):static_cast<void*>(half.data()),n*item,ACL_MEMCPY_HOST_TO_DEVICE),"装载权重");}return out;
}
inline void zero(const T& t){check(aclrtMemset(t->data(),t->size(),0,t->size()),"初始化 tensor");}
struct HostTensor {std::vector<int64_t> shape;aclDataType dtype;std::vector<uint8_t> data;};
using HostState=std::vector<HostTensor>;
inline HostTensor download(const T& t,aclrtStream stream){check(aclrtSynchronizeStream(stream),"快照同步");HostTensor h{t->shape,t->dtype,std::vector<uint8_t>(t->size())};check(aclrtMemcpy(h.data.data(),h.data.size(),t->data(),t->size(),ACL_MEMCPY_DEVICE_TO_HOST),"快照 D2H");return h;}
inline void restore(const T& t,const HostTensor& h){require(t->shape==h.shape&&t->dtype==h.dtype,"快照形状/dtype 不符");upload(t,h.data.data(),h.data.size());}
inline T row_view(const T& t,size_t row){require(row<size_t(t->shape[0]),"状态行越界");auto shape=t->shape;shape[0]=1;return std::make_shared<Tensor>(t->memory,shape,t->dtype,t->stride,t->offset+row*t->stride[0]);}
// 不同步设备的有界主机元数据缓存；只存已经下载的不可变快照。
class PrefixStates {
  using Key=std::vector<int64_t>;struct Item{HostState state;size_t bytes;std::list<Key>::iterator lru;};
  size_t budget_,bytes_=0;std::map<Key,Item> entries_;std::list<Key> lru_;uint64_t saved_=0,restored_=0,evicted_=0;
public:
  explicit PrefixStates(size_t budget):budget_(budget){}
  bool has(const Key& key)const{return entries_.contains(key);}
  void erase(const Key& key){if(auto i=entries_.find(key);i!=entries_.end()){bytes_-=i->second.bytes;lru_.erase(i->second.lru);entries_.erase(i);++evicted_;}}
  void put(const Key& key,HostState state){erase(key);size_t n=0;for(auto& h:state){require(h.data.size()<=SIZE_MAX-n,"快照字节数溢出");n+=h.data.size();}if(!n||!budget_||n>budget_)return;while(n>budget_-bytes_)erase(lru_.front());lru_.push_back(key);entries_.emplace(key,Item{std::move(state),n,std::prev(lru_.end())});bytes_+=n;++saved_;}
  const HostState& get(const Key& key){auto& e=entries_.at(key);lru_.splice(lru_.end(),lru_,e.lru);++restored_;return e.state;}
  Json snapshot()const{return {{"budget_bytes",budget_},{"bytes",bytes_},{"entries",entries_.size()},{"saved",saved_},{"restored",restored_},{"evicted",evicted_}};}
};

struct Net {
  Json config;int h,hq,hkv,dim,layers,rotary;double eps;bool eagle,hybrid;
  std::map<std::string,T> weights;std::vector<T> keys,values,conv,ssm;T embed,head,rope,state_slot;
  Net(const std::map<std::string,WeightView>& source,Json cfg,const Config& c,T shared_embed={},T shared_head={}):config(std::move(cfg)){
    h=config.at("hidden_size");hq=config.at("num_attention_heads");hkv=config.at("num_key_value_heads");dim=config.at("head_dim");layers=config.at("num_hidden_layers");eps=config.at("rms_norm_eps");eagle=config.value("spec_method",std::string())=="eagle3";hybrid=config.value("model_type",std::string()).starts_with("qwen3_5");rotary=hybrid?config.at("rotary_dim").get<int>():dim;
    for(auto& [name,v]:source)if(!name.ends_with(".weight_scale")&&!name.ends_with(".weight_zero_point"))weights[name]=load_weight(source,name,name.ends_with("A_log")||name.ends_with("dt_bias"));
    embed=weights.contains("embed_tokens.weight")?w("embed_tokens.weight"):shared_embed;require(bool(embed),"缺少 embedding");head=weights.contains("lm_head.weight")?w("lm_head.weight"):shared_head?shared_head:embed;
    if(hybrid)require(config.at("mamba_ssm_dtype")=="float32"&&(config.at("output_gate_type")=="silu"||config.at("output_gate_type")=="sigmoid"),"不支持的 GDN 状态类型/门控");
    for(int l=0;l<layers;++l){auto p="layers."+std::to_string(l)+".";
      if(moe_layer(config,l)&&!weights.contains(p+"mlp.experts.gate_up_proj.weight")){
        int e=config.at("num_experts"),m=config.at("moe_intermediate_size");auto gu=tensor({e,2*m,h}),down=tensor({e,h,m});
        for(int j=0;j<e;++j)for(std::string part:{"gate","up","down"}){auto name=p+"mlp.experts."+std::to_string(j)+"."+part+"_proj.weight";auto v=w(name),dst=part=="down"?down:gu;size_t offset=part=="down"?size_t(j)*h*m:size_t(j)*2*m*h+(part=="up"?m*h:0);check(aclrtMemcpy(static_cast<char*>(dst->data())+offset*2,dst->size()-offset*2,v->data(),v->size(),ACL_MEMCPY_DEVICE_TO_DEVICE),"聚合专家");weights.erase(name);}weights[p+"mlp.experts.gate_up_proj.weight"]=gu;weights[p+"mlp.experts.down_proj.weight"]=down;
      }
      bool linear=hybrid&&config.at("layer_types")[l]=="linear_attention";
      if(linear){int nk=config.at("linear_num_key_heads"),nv=config.at("linear_num_value_heads"),dk=config.at("linear_key_head_dim"),dv=config.at("linear_value_head_dim"),kernel=config.at("linear_conv_kernel_dim"),cd=2*nk*dk+nv*dv;T history;if(kernel>1){history=tensor({c.max_num_seqs+1,cd,kernel-1});zero(history);}conv.push_back(history);auto state=tensor({c.max_num_seqs+1,nv,dk,dv},ACL_FLOAT);zero(state);ssm.push_back(state);keys.push_back({});values.push_back({});}
      else{auto k=tensor({(c.num_kvcache_blocks+1)*c.kvcache_block_size,hkv*dim}),v=tensor(k->shape);zero(k);zero(v);keys.push_back(k);values.push_back(v);conv.push_back({});ssm.push_back({});}
    }
    rope=tensor({c.max_model_len,rotary},ACL_FLOAT);std::vector<float> cache(c.max_model_len*rotary);double theta=config.at("rope_theta");for(int64_t p=0;p<c.max_model_len;++p)for(int j=0;j<rotary/2;++j){float f=float(p)/std::pow(float(theta),float(j*2)/rotary);cache[p*rotary+j]=std::cos(f);cache[p*rotary+j+rotary/2]=std::sin(f);}upload(rope,cache.data(),cache.size()*4);
    state_slot=tensor({1},ACL_INT64);int64_t dummy=c.max_num_seqs;upload(state_slot,&dummy,8);
  }
  T w(const std::string& name)const{return weights.at(name);}
  bool recurrent()const{return std::any_of(ssm.begin(),ssm.end(),[](const T& state){return bool(state);});}
  T norm(Program& p,const T& x,const T& gamma,bool gemma=false){auto weight=p.cast(gamma,ACL_FLOAT);if(gemma)weight=p.binary(weight,p.constant(1,ACL_FLOAT),'+');return p.norm(x,weight,eps);}
  T rotate(Program& p,const T& input,const T& positions,int heads){int rows=positions->shape[0];auto x=p.view(p.cast(input,ACL_FLOAT),{rows,heads,dim}),table=p.gather(rope,positions,0);auto cs=p.view(p.narrow(table,1,0,rotary/2),{rows,1,rotary/2}),sn=p.view(p.narrow(table,1,rotary/2,rotary/2),{rows,1,rotary/2});auto a=p.narrow(x,2,0,rotary/2),b=p.narrow(x,2,rotary/2,rotary/2);std::vector<T> parts{p.binary(p.binary(a,cs,'*'),p.binary(b,sn,'*'),'-'),p.binary(p.binary(b,cs,'*'),p.binary(a,sn,'*'),'+')};if(rotary<dim)parts.push_back(p.narrow(x,2,rotary,dim-rotary));return p.cast(p.cat(parts,2),ACL_FLOAT16);}
  T attention(Program& p,int l,const T& q,const T& k,const T& v,const T& positions,const T& slots,const T& page_indices,const T& mask){
    int rows=q->shape[0],context=page_indices->shape[0];p.scatter(keys[l],slots,p.view(rotate(p,k,positions,hkv),{rows,hkv*dim}));p.scatter(values[l],slots,p.view(v,{rows,hkv*dim}));auto query=rotate(p,q,positions,hq);
    std::vector<int64_t> groups(hq);for(int j=0;j<hq;++j)groups[j]=j/(hq/hkv);auto heads=p.indices(groups);auto key=p.gather(p.view(p.gather(keys[l],page_indices,0),{context,hkv,dim}),heads,1),value=p.gather(p.view(p.gather(values[l],page_indices,0),{context,hkv,dim}),heads,1);
    auto score=p.scale(p.cast(p.matmul(p.transpose(query,0,1),p.transpose(p.transpose(key,0,1),1,2)),ACL_FLOAT),1.f/std::sqrt(float(dim)));auto probabilities=p.cast(p.softmax(p.binary(score,p.view(p.cast(mask,ACL_FLOAT),{1,rows,context}),'+')),ACL_FLOAT16);return p.transpose(p.matmul(probabilities,p.transpose(value,0,1)),0,1);
  }
  T mlp(Program& p,const T& x,const std::string& prefix,int lid){
    if(!moe_layer(config,lid)){auto gu=p.cat({p.linear(x,w(prefix+"gate_proj.weight")),p.linear(x,w(prefix+"up_proj.weight"))},1);return p.linear(p.silu_mul(gu),w(prefix+"down_proj.weight"));}
    int rows=x->shape[0],top=config.at("num_experts_per_tok"),width=config.at("moe_intermediate_size");auto [route,ids]=p.topk(p.softmax(p.linear(x,w(prefix+"gate.weight"))),top);if(config.value("norm_topk_prob",true))route=p.binary(route,p.sum(route,1,true),'/');route=p.cast(route,ACL_FLOAT16);T out;
    for(int j=0;j<top;++j){auto choices=p.view(p.narrow(ids,1,j,1),{rows});auto gu=p.gather(w(prefix+"experts.gate_up_proj.weight"),choices,0),down=p.gather(w(prefix+"experts.down_proj.weight"),choices,0);auto activation=p.silu_mul(p.view(p.matmul(gu,p.view(x,{rows,h,1})),{rows,2*width}));auto term=p.binary(p.view(p.matmul(down,p.view(activation,{rows,width,1})),{rows,h}),p.narrow(route,1,j,1),'*');out=out?p.binary(out,term,'+'):term;}
    if(hybrid){auto gu=p.cat({p.linear(x,w(prefix+"shared_expert.gate_proj.weight")),p.linear(x,w(prefix+"shared_expert.up_proj.weight"))},1);out=p.binary(out,p.binary(p.linear(p.silu_mul(gu),w(prefix+"shared_expert.down_proj.weight")),p.unary(p.linear(x,w(prefix+"shared_expert_gate.weight")),"sigmoid"),'*'),'+');}return out;
  }
  T gdn(Program& p,const T& x,int lid,const std::string& prefix){
    int nk=config.at("linear_num_key_heads"),nv=config.at("linear_num_value_heads"),dk=config.at("linear_key_head_dim"),dv=config.at("linear_value_head_dim"),kernel=config.at("linear_conv_kernel_dim"),cd=2*nk*dk+nv*dv;
    auto raw=p.linear(x,w(prefix+"in_proj_qkv.weight")),z=p.view(p.linear(x,w(prefix+"in_proj_z.weight")),{1,nv,dv});auto window=p.view(raw,{1,cd,1});if(kernel>1)window=p.cat({p.gather(conv[lid],state_slot,0),window},2);
    auto convolved=p.unary(p.cast(p.sum(p.binary(p.cast(window,ACL_FLOAT),p.view(p.cast(w(prefix+"conv1d.weight"),ACL_FLOAT),{1,cd,kernel}),'*'),2,false),ACL_FLOAT16),"silu");if(kernel>1)p.scatter(conv[lid],state_slot,p.narrow(window,2,1,kernel-1));
    std::vector<int64_t> repeat(nv);for(int j=0;j<nv;++j)repeat[j]=j/(nv/nk);auto groups=p.indices(repeat);
    auto q=p.gather(p.view(p.narrow(convolved,1,0,nk*dk),{1,nk,dk}),groups,1),k=p.gather(p.view(p.narrow(convolved,1,nk*dk,nk*dk),{1,nk,dk}),groups,1),v=p.view(p.narrow(convolved,1,2*nk*dk,nv*dv),{1,nv,dv});q=p.scale(p.l2(q,1e-6),1.f/std::sqrt(float(dk)));k=p.l2(k,1e-6);
    auto a=p.cast(p.linear(x,w(prefix+"in_proj_a.weight")),ACL_FLOAT),b=p.linear(x,w(prefix+"in_proj_b.weight"));auto decay=p.unary(p.scale(p.binary(p.unary(w(prefix+"A_log"),"exp"),p.unary(p.binary(a,w(prefix+"dt_bias"),'+'),"softplus"),'*'),-1),"exp"),beta=p.cast(p.unary(b,"sigmoid"),ACL_FLOAT);
    auto state=p.binary(p.gather(ssm[lid],state_slot,0),p.view(decay,{1,nv,1,1}),'*');auto memory=p.sum(p.binary(state,p.view(k,{1,nv,dk,1}),'*'),2,false);auto delta=p.binary(p.binary(p.cast(v,ACL_FLOAT),memory,'-'),p.view(beta,{1,nv,1}),'*');state=p.binary(state,p.binary(p.view(k,{1,nv,dk,1}),p.view(delta,{1,nv,1,dv}),'*'),'+');p.scatter(ssm[lid],state_slot,state);
    auto core=p.cast(p.sum(p.binary(state,p.view(q,{1,nv,dk,1}),'*'),2,false),ACL_FLOAT16),fp=p.cast(core,ACL_FLOAT);auto unit=p.binary(fp,p.unary(p.binary(p.scale(p.sum(p.binary(fp,fp,'*'),2,true),1.f/dv),p.constant(eps,ACL_FLOAT),'+'),"rsqrt"),'*');auto normalized=p.binary(p.cast(unit,ACL_FLOAT16),w(prefix+"norm.weight"),'*');auto gate=p.unary(p.cast(z,ACL_FLOAT),config.at("output_gate_type").get<std::string>());return p.linear(p.view(p.cast(p.binary(p.cast(normalized,ACL_FLOAT),gate,'*'),ACL_FLOAT16),{1,nv*dv}),w(prefix+"out_proj.weight"));
  }
  std::pair<T,T> forward(Program& p,const T& ids,const T& positions,const T& slots,const T& pages,const T& mask,const std::vector<int>& aux_layers,const T& seed={}){
    int rows=ids->shape[0];require(!hybrid||rows==1,"GDN 参考路径要求单行顺序执行");auto hidden=p.gather(embed,ids,0);std::map<int,T> captured;
    if(eagle){require(bool(seed),"EAGLE 缺少 seed");auto prefix=std::string("midlayer.");auto input=p.cat({norm(p,hidden,w(prefix+"input_layernorm.weight")),norm(p,seed,w(prefix+"hidden_norm.weight"))},1);auto q=p.view(p.linear(input,w(prefix+"self_attn.q_proj.weight")),{rows,hq,dim}),k=p.view(p.linear(input,w(prefix+"self_attn.k_proj.weight")),{rows,hkv,dim}),v=p.view(p.linear(input,w(prefix+"self_attn.v_proj.weight")),{rows,hkv,dim});hidden=p.binary(seed,p.linear(p.view(attention(p,0,q,k,v,positions,slots,pages,mask),{rows,hq*dim}),w(prefix+"self_attn.o_proj.weight")),'+');hidden=p.binary(hidden,mlp(p,norm(p,hidden,w(prefix+"post_attention_layernorm.weight")),prefix+"mlp.",0),'+');return {hidden,{}};}
    T residual;
    for(int l=0;l<layers;++l){auto prefix="layers."+std::to_string(l)+".";if(hybrid){if(std::find(aux_layers.begin(),aux_layers.end(),l)!=aux_layers.end())captured[l]=hidden;auto normalized=norm(p,hidden,w(prefix+"input_layernorm.weight"),true);T value;
        if(ssm[l])value=gdn(p,normalized,l,prefix+"linear_attn.");else{auto a=prefix+"self_attn.";auto qgate=p.view(p.linear(normalized,w(a+"q_proj.weight")),{rows,hq,2*dim});auto q=norm(p,p.narrow(qgate,2,0,dim),w(a+"q_norm.weight"),true),k=norm(p,p.view(p.linear(normalized,w(a+"k_proj.weight")),{rows,hkv,dim}),w(a+"k_norm.weight"),true),v=p.view(p.linear(normalized,w(a+"v_proj.weight")),{rows,hkv,dim});value=p.linear(p.view(p.binary(attention(p,l,q,k,v,positions,slots,pages,mask),p.unary(p.narrow(qgate,2,dim,dim),"sigmoid"),'*'),{rows,hq*dim}),w(a+"o_proj.weight"));}
        hidden=p.binary(hidden,value,'+');hidden=p.binary(hidden,mlp(p,norm(p,hidden,w(prefix+"post_attention_layernorm.weight"),true),prefix+"mlp.",l),'+');
      }else{auto sum=residual?p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+'):p.cast(hidden,ACL_FLOAT);if(std::find(aux_layers.begin(),aux_layers.end(),l)!=aux_layers.end())captured[l]=p.cast(sum,ACL_FLOAT16);residual=p.cast(sum,ACL_FLOAT16);auto normalized=norm(p,sum,w(prefix+"input_layernorm.weight"));auto a=prefix+"self_attn.";auto q=norm(p,p.view(p.linear(normalized,w(a+"q_proj.weight")),{rows,hq,dim}),w(a+"q_norm.weight")),k=norm(p,p.view(p.linear(normalized,w(a+"k_proj.weight")),{rows,hkv,dim}),w(a+"k_norm.weight")),v=p.view(p.linear(normalized,w(a+"v_proj.weight")),{rows,hkv,dim});hidden=p.linear(p.view(attention(p,l,q,k,v,positions,slots,pages,mask),{rows,hq*dim}),w(a+"o_proj.weight"));sum=p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+');residual=p.cast(sum,ACL_FLOAT16);hidden=mlp(p,norm(p,sum,w(prefix+"post_attention_layernorm.weight")),prefix+"mlp.",l);}
    }
    if(!hybrid)hidden=p.binary(p.cast(hidden,ACL_FLOAT),p.cast(residual,ACL_FLOAT),'+');if(std::find(aux_layers.begin(),aux_layers.end(),layers)!=aux_layers.end())captured[layers]=p.cast(hidden,ACL_FLOAT16);std::vector<T> aux;for(int id:aux_layers)aux.push_back(captured.at(id));return {norm(p,hidden,w("norm.weight"),hybrid),aux.empty()?T{}:p.cat(aux,1)};
  }
  void project(Program& p,const T& aux,const T& positions,const T& slots){int rows=positions->shape[0];auto x=norm(p,p.linear(aux,w("fc.weight")),w("hidden_norm.weight"));for(int l=0;l<layers;++l){auto a="layers."+std::to_string(l)+".self_attn.";auto k=norm(p,p.view(p.linear(x,w(a+"k_proj.weight")),{rows,hkv,dim}),w(a+"k_norm.weight"));p.scatter(keys[l],slots,p.view(rotate(p,k,positions,hkv),{rows,hkv*dim}));p.scatter(values[l],slots,p.linear(x,w(a+"v_proj.weight")));}}
  HostState save(int slot,aclrtStream stream){HostState out;for(int l=0;l<layers;++l)if(ssm[l]){if(conv[l])out.push_back(download(row_view(conv[l],slot),stream));out.push_back(download(row_view(ssm[l],slot),stream));}return out;}
  void load(int slot,const HostState& saved){size_t j=0;for(int l=0;l<layers;++l)if(ssm[l]){if(conv[l])restore(row_view(conv[l],slot),saved.at(j++));restore(row_view(ssm[l],slot),saved.at(j++));}require(j==saved.size(),"GDN 快照数目错误");}
  void clear(int slot){for(int l=0;l<layers;++l)if(ssm[l]){if(conv[l])zero(row_view(conv[l],slot));zero(row_view(ssm[l],slot));}}
};
}
