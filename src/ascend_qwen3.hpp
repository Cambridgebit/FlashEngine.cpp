#pragma once
// Qwen3 / DFlash 的纯原生执行：与 Python 相同的打包投影、NZ KV 和 ATB Attention。
#include "ascend_net.hpp"
#include "fe/model.hpp"
#include <iostream>
#include <set>

namespace fe::ascend {
inline T physical_nz(std::vector<int64_t> shape){auto t=tensor(std::move(shape));t->atb_format=ACL_FORMAT_FRACTAL_NZ;zero(t);return t;}

inline std::pair<T,T> qwen_rope_cache(int positions,int dim,float theta,aclrtStream stream){
  // Python 在 NPU FP32 上计算 pow、倒数、外积和三角函数，最后转 FP16。
  // 主机 libm 的近似值会跨越 FP16 舍入边界，不能用数学等价替代逐算子对齐。
  Program p;auto exponent=p.out({dim/2},ACL_FLOAT),position=p.out({positions},ACL_FLOAT),power=p.out({dim/2},ACL_FLOAT);std::vector<float> ex(dim/2),pos(positions);for(int j=0;j<dim/2;++j)ex[j]=float(j*2)/dim;std::iota(pos.begin(),pos.end(),0.f);upload(exponent,ex.data(),ex.size()*4);upload(position,pos.data(),pos.size()*4);
  p.add(aclnnPowScalarTensorGetWorkspaceSize,aclnnPowScalarTensor,p.scalar(theta),exponent->handle,power->handle);auto inv=p.binary(p.constant(1,ACL_FLOAT),power,'/'),freq=p.binary(p.view(position,{positions,1}),p.view(inv,{1,dim/2}),'*'),cs=p.out({positions,dim/2},ACL_FLOAT),sn=p.out({positions,dim/2},ACL_FLOAT);p.add(aclnnCosGetWorkspaceSize,aclnnCos,freq->handle,cs->handle);p.add(aclnnSinGetWorkspaceSize,aclnnSin,freq->handle,sn->handle);
  cs=p.cast(cs,ACL_FLOAT16);sn=p.cast(sn,ACL_FLOAT16);auto cos=p.cat({cs,cs},1),sin=p.cat({sn,sn},1);p.finish();p.eager(stream);check(aclrtSynchronizeStream(stream),"NPU RoPE 常量生成");return {cos,sin};
}

struct QwenNet {
  Net net;T cos_cache,sin_cache,fused_context_kv;bool draft;Config config;
  QwenNet(const std::map<std::string,WeightView>& source,const Json& model,const Config& c,aclrtStream stream,T embed={},T head={}):net(source,model,c,embed,head),draft(bool(embed)),config(c){
    require(!net.hybrid&&!net.eagle,"批量 ATB 网络仅接受 Qwen3 / DFlash");
    require(net.dim%16==0&&c.kvcache_block_size%16==0&&c.kvcache_block_size<=128&&c.kvcache_block_size*net.dim<=16384,"310P NZ KV 页或 head_dim 不受支持");
    auto convert=[&](const std::vector<T>& parts){Program p;auto nd=parts.size()==1?parts[0]:p.cat(parts,0);auto out=p.nz(nd);p.finish();p.eager(stream);check(aclrtSynchronizeStream(stream),"NZ 权重预转换");return out;};
    if(draft){std::vector<T> parts;for(int l=0;l<net.layers;++l){auto p="layers."+std::to_string(l)+".self_attn.";parts.push_back(net.w(p+"k_proj.weight"));parts.push_back(net.w(p+"v_proj.weight"));}fused_context_kv=convert(parts);}
    for(int l=0;l<net.layers;++l){auto p="layers."+std::to_string(l)+".",a=p+"self_attn.";
      net.weights[a+"qkv_proj.weight"]=convert({net.w(a+"q_proj.weight"),net.w(a+"k_proj.weight"),net.w(a+"v_proj.weight")});for(auto part:{"q","k","v"})net.weights.erase(a+part+"_proj.weight");
      net.weights[a+"o_proj.weight"]=convert({net.w(a+"o_proj.weight")});
      if(!moe_layer(net.config,l)){auto m=p+"mlp.";net.weights[m+"gate_up_proj.weight"]=convert({net.w(m+"gate_proj.weight"),net.w(m+"up_proj.weight")});net.weights.erase(m+"gate_proj.weight");net.weights.erase(m+"up_proj.weight");net.weights[m+"down_proj.weight"]=convert({net.w(m+"down_proj.weight")});}
      auto shape=std::vector<int64_t>{c.num_kvcache_blocks+1,net.hkv*net.dim/16,c.kvcache_block_size,16};net.keys[l]=physical_nz(shape);net.values[l]=physical_nz(shape);
      if((l+1)%8==0||l+1==net.layers)std::cerr<<"[INFO ] "<<(draft?"DFlash":"目标")<<" NZ 权重 "<<l+1<<'/'<<net.layers<<'\n';
    }
    if(net.head->atb_format!=ACL_FORMAT_FRACTAL_NZ){std::cerr<<"[INFO ] 转换 NZ LM-head\n";net.head=convert({net.head});std::cerr<<"[INFO ] NZ LM-head 就绪\n";}
    if(net.weights.contains("lm_head.weight"))net.weights["lm_head.weight"]=net.head;
    if(draft)net.weights["fc.weight"]=convert({net.w("fc.weight")});
    auto cache=qwen_rope_cache(c.max_model_len,net.dim,net.config.at("rope_theta").get<float>(),stream);cos_cache=cache.first;sin_cache=cache.second;net.rope.reset();
  }
  T w(const std::string& name)const{return net.w(name);}
};

struct QwenMeta {
  int n,rows,width;std::vector<int32_t> qlens;std::shared_ptr<std::vector<int32_t>> host_qlens;
  T positions,slots,table,lens,mask;bool cold,noncausal;
  QwenMeta(Program& p,const Config& c,std::vector<int32_t> queries,bool is_cold=false,bool block=false):n(queries.size()),rows(std::accumulate(queries.begin(),queries.end(),0)),width((c.max_model_len+c.kvcache_block_size-1)/c.kvcache_block_size),qlens(std::move(queries)),host_qlens(std::make_shared<std::vector<int32_t>>(qlens)),cold(is_cold),noncausal(block){
    positions=p.out({rows},ACL_INT64);slots=p.out({rows},ACL_INT32);table=p.out({n,width},ACL_INT32);lens=p.out({n},ACL_INT32);
    int columns=cold?((rows+15)/16*16):((c.max_model_len+15)/16*16),mask_rows=(rows+15)/16*16;if(cold)host_qlens->back()+=columns-rows;mask=p.keep(physical_nz({1,columns/16,mask_rows,16}));for(auto& t:{positions,slots,table,lens})zero(t);
  }
  void prepare(const std::vector<Seq>& seqs,const Config& c,const std::vector<int64_t>& starts,const std::vector<int32_t>& actual){
    require(seqs.size()==size_t(n)&&starts.size()==size_t(n)&&actual.size()==size_t(n),"ATB 批元数据数量错误");std::vector<int64_t> pos(rows);std::vector<int32_t> ids(rows),tables(n*width,c.num_kvcache_blocks),lengths(n);int offset=0;int columns=mask->shape[1]*16,mask_rows=mask->shape[2];std::vector<uint16_t> bias(mask_rows*columns,0);
    if(cold)for(int row=0;row<mask_rows;++row)for(int t=row+1;t<columns;++t)bias[(t/16*mask_rows+row)*16+t%16]=fp16(-INFINITY);
    for(int i=0;i<n;++i){auto& s=seqs[i];require(actual[i]>0&&actual[i]<=qlens[i],"ATB 实际查询长度错误");lengths[i]=starts[i]+actual[i];if(cold)lengths[i]=qlens[i];
      if(s)for(size_t j=0;j<s->blocks.size()&&j<size_t(width);++j)tables[i*width+j]=s->blocks[j];
      for(int j=0;j<qlens[i];++j){int row=offset+j;int64_t logical=starts[i]+j;pos[row]=std::min<int64_t>(logical,c.max_model_len-1);ids[row]=s&&j<actual[i]?s->blocks.at(logical/c.kvcache_block_size)*c.kvcache_block_size+logical%c.kvcache_block_size:c.num_kvcache_blocks*c.kvcache_block_size+j%c.kvcache_block_size;
        if(!cold&&!noncausal)for(int t=0;t<columns;++t)bias[(t/16*mask_rows+row)*16+t%16]=fp16(t<=logical?0.f:-INFINITY);
      }offset+=qlens[i];
    }
    upload(positions,pos.data(),pos.size()*8);upload(slots,ids.data(),ids.size()*4);upload(table,tables.data(),tables.size()*4);upload(lens,lengths.data(),lengths.size()*4);upload(mask,bias.data(),bias.size()*2);
  }
};

inline std::pair<T,T> rotary_pair(Program& p,const T& query,const T& key,const T& cos,const T& sin,const T& length){
  // 与 Python npu_apply_rotary_pos_emb 使用同一个公开 ACLNN V2 算子。
  // 分离输入副本，保留共享 Q/K 的上下文投影语义，避免原地操作污染源张量。
  (void)length;int64_t rows=query->shape[0],dim=cos->shape.back();auto q=p.out(query->shape),k=p.out(key->shape);p.copy(q,query);p.copy(k,key);
  auto q4=p.view(q,{1,rows,elements(q->shape)/rows/dim,dim}),k4=p.view(k,{1,rows,elements(k->shape)/rows/dim,dim}),c4=p.view(cos,{1,rows,1,dim}),s4=p.view(sin,{1,rows,1,dim});
  static char mode[]="half";p.add(aclnnApplyRotaryPosEmbV2GetWorkspaceSize,aclnnApplyRotaryPosEmbV2,q4->handle,k4->handle,c4->handle,s4->handle,int64_t(1),mode);return {q,k};
}
struct QwenRope {
  T cos,sin,length;
  QwenRope(Program& p,const QwenNet& net,const T& positions){cos=p.gather(net.cos_cache,positions,0);sin=p.gather(net.sin_cache,positions,0);length=p.cast(p.indices({positions->shape[0]}),ACL_INT32);}
  std::pair<T,T> apply(Program& p,const T& query,const T& key){return rotary_pair(p,query,key,cos,sin,length);}
};
inline void cache_kv(Program& p,const T& k,const T& v,const T& kc,const T& vc,const T& slots){atb::infer::ReshapeAndCacheParam param;param.kvCacheCfg=atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;p.atb(param,{p.dense(k),p.dense(v),kc,vc,slots},{kc,vc});}

inline T qwen_forward(Program& p,QwenNet& model,QwenMeta& meta,const T& ids,const std::vector<int>& aux_layers,T& aux){
  auto& net=model.net;int rows=meta.rows;auto hidden=p.gather(net.embed,ids,0);T residual;std::vector<T> captured;QwenRope rope(p,model,meta.positions);
  for(int l=0;l<net.layers;++l){auto prefix="layers."+std::to_string(l)+".",a=prefix+"self_attn.";
    if(std::find(aux_layers.begin(),aux_layers.end(),l)!=aux_layers.end())captured.push_back(residual?p.binary(hidden,residual,'+'):hidden);
    T normalized;if(residual){auto pair=p.add_norm(hidden,residual,net.w(prefix+"input_layernorm.weight"),net.eps);normalized=pair.first;residual=pair.second;}else{residual=hidden;normalized=p.fast_norm(hidden,net.w(prefix+"input_layernorm.weight"),net.eps);}
    auto qkv=p.linear(normalized,net.w(a+"qkv_proj.weight"));int qw=net.hq*net.dim,kw=net.hkv*net.dim;
    auto q=p.fast_norm(p.view(p.narrow(qkv,1,0,qw),{rows,net.hq,net.dim}),net.w(a+"q_norm.weight"),net.eps),k=p.fast_norm(p.view(p.narrow(qkv,1,qw,kw),{rows,net.hkv,net.dim}),net.w(a+"k_norm.weight"),net.eps),v=p.view(p.narrow(qkv,1,qw+kw,kw),{rows,net.hkv,net.dim});
    auto rotated=rope.apply(p,q,k);q=rotated.first;k=rotated.second;v=p.dense(v);cache_kv(p,k,v,net.keys[l],net.values[l],meta.slots);auto output=p.out({rows,net.hq,net.dim});
    if(meta.cold){
      // Python 仅在 Flash Attention 内部补齐到 16；投影、归一化、MLP 保留真实行数。
      int aligned=(rows+15)/16*16;auto pad=[&](const T& x){if(aligned==rows)return x;auto shape=x->shape;shape[0]=aligned;auto y=p.out(shape);zero(y);p.copy(p.narrow(y,0,0,rows),x);return y;};auto q_a=pad(q),k_a=pad(k),v_a=pad(v),out_a=aligned==rows?output:p.out({aligned,net.hq,net.dim});
      atb::infer::SelfAttentionParam param;param.calcType=atb::infer::SelfAttentionParam::PA_ENCODER;param.maskType=atb::infer::SelfAttentionParam::MASK_TYPE_NORM;param.headNum=net.hq;param.kvHeadNum=net.hkv;param.qkScale=1.f/std::sqrt(float(net.dim));p.atb(param,{q_a,k_a,v_a,meta.mask},{out_a},meta.host_qlens);if(aligned!=rows)output=p.narrow(out_a,0,0,rows);
    }
    else{atb::infer::PagedAttentionParam param;param.headNum=net.hq;param.kvHeadNum=net.hkv;param.qkScale=1.f/std::sqrt(float(net.dim));std::vector<T> inputs{q,net.keys[l],net.values[l],meta.table,meta.lens};
      bool single=std::all_of(meta.qlens.begin(),meta.qlens.end(),[](int v){return v==1;});if(!single){param.maskType=atb::infer::PagedAttentionParam::MASK_TYPE_SPEC;param.calcType=atb::infer::PagedAttentionParam::CALC_TYPE_SPEC;inputs.push_back(meta.mask);}p.atb(param,inputs,{output},single?nullptr:meta.host_qlens);
    }
    hidden=p.linear(p.view(output,{rows,qw}),net.w(a+"o_proj.weight"));auto pair=p.add_norm(hidden,residual,net.w(prefix+"post_attention_layernorm.weight"),net.eps);residual=pair.second;
    if(moe_layer(net.config,l))hidden=net.mlp(p,pair.first,prefix+"mlp.",l);else hidden=p.linear(p.swiglu(p.linear(pair.first,net.w(prefix+"mlp.gate_up_proj.weight"))),net.w(prefix+"mlp.down_proj.weight"));
  }
  if(!captured.empty())aux=p.cat(captured,1);return p.add_norm(hidden,residual,net.w("norm.weight"),net.eps).first;
}

inline void qwen_project(Program& p,QwenNet& draft,const T& auxiliary,const T& positions,const T& slots){
  auto& net=draft.net;int rows=positions->shape[0],kw=net.hkv*net.dim;auto x=p.fast_norm(p.linear(auxiliary,net.w("fc.weight")),net.w("hidden_norm.weight"),net.eps),kv=p.linear(x,draft.fused_context_kv);QwenRope rope(p,draft,positions);
  for(int l=0;l<net.layers;++l){auto a="layers."+std::to_string(l)+".self_attn.";auto k=p.fast_norm(p.view(p.narrow(kv,1,l*2*kw,kw),{rows,net.hkv,net.dim}),net.w(a+"k_norm.weight"),net.eps),v=p.view(p.narrow(kv,1,l*2*kw+kw,kw),{rows,net.hkv,net.dim});auto rotated=rope.apply(p,k,k);cache_kv(p,rotated.first,v,net.keys[l],net.values[l],slots);}
}

struct QwenEntry {
  Program program;QwenMeta meta;T ids,last,output,logits;bool prefill;std::vector<int> aux_layers;
  QwenEntry(QwenNet& target,QwenNet* draft,const Config& c,std::vector<int32_t> queries,bool cold,bool is_prefill,const std::vector<int>& layers):meta(program,c,std::move(queries),cold),prefill(is_prefill),aux_layers(layers){
    ids=program.out({meta.rows},ACL_INT64);last=program.out({meta.n},ACL_INT64);zero(ids);zero(last);T aux;auto hidden=qwen_forward(program,target,meta,ids,layers,aux);
    logits=program.linear(program.gather(hidden,last,0),target.net.head);output=program.argmax(logits);
    if(draft){int offset=0;for(auto count:meta.qlens){qwen_project(program,*draft,program.narrow(aux,0,offset,count),program.narrow(meta.positions,0,offset,count),program.narrow(meta.slots,0,offset,count));offset+=count;}}program.finish();
  }
  void prepare(const std::vector<Seq>& seqs,const Config& c){
    std::vector<int64_t> input(meta.rows,0),starts(meta.n),indices(meta.n);std::vector<int32_t> actual(meta.n);int offset=0;
    auto padded=seqs;padded.resize(meta.n);for(int i=0;i<meta.n;++i){auto& s=padded[i];starts[i]=s?(prefill?s->cached:s->tokens.size()-1):0;actual[i]=s?(prefill?s->scheduled:1):1;if(s)for(int j=0;j<actual[i];++j)input[offset+j]=s->tokens.at(starts[i]+j);indices[i]=offset+actual[i]-1;offset+=meta.qlens[i];}
    meta.prepare(padded,c,starts,actual);upload(ids,input.data(),input.size()*8);upload(last,indices.data(),indices.size()*8);
  }
};

inline std::pair<T,T> greedy_device_commit(Program& program,const T& proposed,const T& verified,const T& input_slots,int64_t dummy_begin){
  int n=proposed->shape[0],k=proposed->shape[1],rows=k+1;
  // 首个不匹配位置等于接受长度；与 Python 的 where + min 等价，避免 310P 的 AI-CPU cumprod。
  std::vector<int64_t> indices(k),all(rows),dummy(rows);std::iota(indices.begin(),indices.end(),0);std::iota(all.begin(),all.end(),0);std::iota(dummy.begin(),dummy.end(),dummy_begin);
  auto ix=program.cast(program.view(program.indices(indices),{1,k}),ACL_FLOAT),match=program.cast(program.compare(proposed,program.narrow(verified,1,0,k),true),ACL_FLOAT);
  auto cost=program.binary(program.scale(match,float(k)),program.binary(program.binary(program.constant(1,ACL_FLOAT),match,'-'),ix,'*'),'+');auto count=program.binary(program.minimum(cost),program.constant(1,ACL_FLOAT),'+');
  auto valid=program.cast(program.compare(program.cast(program.view(program.indices(all),{1,rows}),ACL_FLOAT),count,false),ACL_INT32);auto one=program.cast(program.constant(1,ACL_INT64),ACL_INT32),dummy_slots=program.cast(program.view(program.indices(dummy),{1,rows}),ACL_INT32);
  auto slots=program.view(program.binary(program.binary(valid,program.view(input_slots,{n,rows}),'*'),program.binary(program.binary(one,valid,'-'),dummy_slots,'*'),'+'),{n*rows});return {program.cast(count,ACL_INT64),slots};
}

struct DFlashEntry {
  Program program;QwenMeta target_meta,draft_meta;T anchors,output,draft_logits,verify_logits;int n,k,rows;
  DFlashEntry(QwenNet& target,QwenNet& draft,const Config& c,int batch,const std::vector<int>& aux_layers):target_meta(program,c,std::vector<int32_t>(batch,c.spec_num_tokens+1)),draft_meta(program,c,std::vector<int32_t>(batch,c.spec_num_tokens+1),false,true),n(batch),k(c.spec_num_tokens),rows(k+1){
    anchors=program.out({n,1},ACL_INT64);zero(anchors);std::vector<int64_t> masks(n*k,draft.net.config.at("mask_token_id"));auto masked=program.view(program.indices(masks),{n,k}),draft_input=program.view(program.cat({anchors,masked},1),{n*rows});T unused;
    auto draft_hidden=qwen_forward(program,draft,draft_meta,draft_input,{},unused);auto selected=program.view(program.narrow(program.view(draft_hidden,{n,rows,draft.net.h}),1,1,k),{n*k,draft.net.h});
    draft_logits=program.linear(selected,draft.net.head);auto proposed=program.argmax(draft_logits);
    if(draft.net.weights.contains("d2t"))proposed=program.binary(proposed,program.view(program.gather(draft.net.w("d2t"),proposed,0),proposed->shape),'+');
    proposed=program.view(proposed,{n,k});auto verify_input=program.view(program.cat({anchors,proposed},1),{n*rows});T aux;auto hidden=qwen_forward(program,target,target_meta,verify_input,aux_layers,aux);verify_logits=program.linear(hidden,target.net.head);auto verified=program.view(program.argmax(verify_logits),{n,rows});
    auto commit=greedy_device_commit(program,proposed,verified,target_meta.slots,int64_t(c.num_kvcache_blocks)*c.kvcache_block_size);
    qwen_project(program,draft,aux,target_meta.positions,commit.second);
    // 每轮一次同步读回；辅助状态和接受掩码全程留在设备。
    output=program.cat({proposed,verified,commit.first},1);program.finish();
    std::vector<Seq> empty(n);std::vector<int64_t> starts(n,0);std::vector<int32_t> actual(n,rows);target_meta.prepare(empty,c,starts,actual);draft_meta.prepare(empty,c,starts,actual);
  }
  void prepare(const std::vector<Seq>& seqs,const Config& c){std::vector<int64_t> ids(n,0),starts(n,0);std::vector<int32_t> actual(n,rows);auto padded=seqs;padded.resize(n);for(size_t i=0;i<seqs.size();++i){ids[i]=seqs[i]->tokens.back();starts[i]=seqs[i]->tokens.size()-1;}target_meta.prepare(padded,c,starts,actual);draft_meta.prepare(padded,c,starts,actual);upload(anchors,ids.data(),ids.size()*8);}
};

// 仅 --logits 诊断读回；常规生成仍保持每轮一次 token 读回。
inline Json qwen_top_logits(const T& logits,int offset,int rows,const std::string& stage,aclrtStream stream){
  auto host=download(logits,stream);require(host.dtype==ACL_FLOAT16,"logits 诊断要求 FP16");auto data=reinterpret_cast<const uint16_t*>(host.data.data());int64_t width=logits->shape[1];Json ids=Json::array(),values=Json::array();std::vector<int64_t> order(width);
  auto value=[&](int row,int64_t col){uint16_t bits=data[(offset+row)*width+col];int e=(bits>>10)&31;float v=std::ldexp(float(bits&1023),-24);if(e==31)v=(bits&1023)?NAN:INFINITY;else if(e)v=std::ldexp(float(1024+(bits&1023)),e-25);return bits&0x8000?-v:v;};
  for(int row=0;row<rows;++row){std::iota(order.begin(),order.end(),0);auto count=std::min<int64_t>(8,width);std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](int64_t a,int64_t b){auto x=value(row,a),y=value(row,b);return x>y||(x==y&&a<b);});std::vector<int64_t> top(order.begin(),order.begin()+count);std::vector<float> score;for(auto col:top)score.push_back(value(row,col));ids.push_back(top);values.push_back(score);}
  return {{"stage",stage},{"ids",ids},{"values",values}};
}
class QwenAscendModel final:public DeviceModel {
  Config config_;Runtime runtime_;std::unique_ptr<QwenNet> target_,draft_;std::vector<int> aux_layers_;std::set<uint64_t> active_;
  std::map<std::vector<int32_t>,std::unique_ptr<QwenEntry>> cold_prefill_,paged_prefill_;std::map<int,std::unique_ptr<QwenEntry>> decode_entries_,limited_entries_;std::map<int,std::unique_ptr<DFlashEntry>> spec_entries_;
  std::list<std::pair<bool,std::vector<int32_t>>> prefill_lru_;
  uint64_t decode_=0,spec_=0,prefill_=0,eager_=0,capture_failed_=0,verify_rows_=0,projected_=0,prefill_evicted_=0;
  int bucket(int n)const{for(auto size:fe::graph_buckets(config_.max_num_seqs))if(size>=n)return int(size);throw std::runtime_error("请求批次超过 Graph 桶上限");}
  void capture(Program& p){if(config_.enforce_eager)return;try{p.capture(runtime_.stream);}catch(const std::exception& e){++capture_failed_;std::cerr<<"[WARN] ACL Graph 捕获失败，使用同构 eager："<<e.what()<<'\n';}}
  QwenEntry& standard_entry(const std::vector<Seq>& seqs,bool prefill){
    int n=bucket(seqs.size());if(!prefill){auto& e=decode_entries_[n];if(!e){e=std::make_unique<QwenEntry>(*target_,draft_.get(),config_,std::vector<int32_t>(n,1),false,false,aux_layers_);e->prepare(seqs,config_);capture(e->program);}return *e;}
    bool cold=std::all_of(seqs.begin(),seqs.end(),[](auto& s){return s->cached==0;});std::vector<int32_t> queries;for(auto& s:seqs)queries.push_back(s->scheduled);
    auto& cache=cold?cold_prefill_:paged_prefill_;auto found=cache.find(queries);auto key=std::make_pair(cold,queries);
    if(found==cache.end()){
      // 逐形状预编译不能随服务请求无限增长；prefill 没有捕获图，读回同步后可安全驱逐。
      if(prefill_lru_.size()>=4){auto old=prefill_lru_.back();prefill_lru_.pop_back();(old.first?cold_prefill_:paged_prefill_).erase(old.second);++prefill_evicted_;}
      found=cache.emplace(queries,std::make_unique<QwenEntry>(*target_,draft_.get(),config_,queries,cold,true,aux_layers_)).first;
    }
    prefill_lru_.remove(key);prefill_lru_.push_front(key);return *found->second;
  }
  ModelOutput standard(const std::vector<Seq>& seqs,bool prefill){auto& e=standard_entry(seqs,prefill);e.prepare(seqs,config_);if(!prefill&&e.program.graph){e.program.run(runtime_.stream);++decode_;}else{e.program.eager(runtime_.stream);++eager_;}auto values=read_ids(e.output,runtime_.stream);ModelOutput out;
    for(size_t i=0;i<seqs.size();++i){auto& s=seqs[i];bool complete=!prefill||s->cached+s->scheduled==s->tokens.size();out.tokens.push_back(complete?std::vector<int64_t>{values[i]}:std::vector<int64_t>{});if(prefill)prefill_+=s->scheduled;if(draft_)projected_+=prefill?s->scheduled:1;if(complete&&config_.trace_logits&&!draft_){auto sample=qwen_top_logits(e.logits,i,1,"target",runtime_.stream);out.samples.push_back({{"token_index",s->tokens.size()-s->prompt_size},{"ids",sample["ids"][0]},{"values",sample["values"][0]}});}}return out;
  }
  ModelOutput limited(const std::vector<Seq>& seqs){
    // 与 Python 的 context_limited_spec_round 相同：实际批大小、eager，只消费 anchor。
    int n=seqs.size();auto& e=limited_entries_[n];if(!e)e=std::make_unique<QwenEntry>(*target_,draft_.get(),config_,std::vector<int32_t>(n,1),false,false,aux_layers_);e->prepare(seqs,config_);e->program.eager(runtime_.stream);++eager_;auto ids=read_ids(e->output,runtime_.stream);ModelOutput out;for(auto id:ids)out.tokens.push_back({id});return out;
  }
public:
  QwenAscendModel(const Bundle& bundle,const Config& config):config_(config){
    require(config.dtype=="float16","ATB Qwen3 需要 FP16");require(config.max_model_len<=bundle.config["target"]["max_position_embeddings"].get<int64_t>(),"模型位置上限不足");std::cerr<<"[INFO ] 加载 ATB Qwen3 目标权重\n";target_=std::make_unique<QwenNet>(bundle.target_weights,bundle.config.at("target"),config,runtime_.stream);
    if(!bundle.config.at("draft").is_null()){auto dc=bundle.config.at("draft");config_.spec_num_tokens=dc.at("num_tokens");require(config_.spec_num_tokens>0&&config_.spec_num_tokens+1<=config_.kvcache_block_size,"DFlash 草稿长度越过 dummy 页");for(int id:dc.at("target_layer_ids").get<std::vector<int>>())aux_layers_.push_back(id+1);std::cerr<<"[INFO ] 加载 DFlash 权重与批量验证程序\n";draft_=std::make_unique<QwenNet>(bundle.draft_weights,dc,config_,runtime_.stream,target_->net.embed,target_->net.head);auto& e=spec_entries_[1];e=std::make_unique<DFlashEntry>(*target_,*draft_,config_,1,aux_layers_);capture(e->program);}
    // 先准备常用 prefill / decode 形状，避免服务首请求承担算子一阶段准备成本。
    int actual=std::min<int64_t>({32,config_.max_num_batched_tokens,config_.max_model_len-1}),queries=actual;auto warm=std::make_shared<Sequence>(0,std::vector<int64_t>(actual,0),1);warm->scheduled=actual;warm->blocks.resize((config_.max_model_len+config_.kvcache_block_size-1)/config_.kvcache_block_size);std::iota(warm->blocks.begin(),warm->blocks.end(),0);auto entry=std::make_unique<QwenEntry>(*target_,draft_.get(),config_,std::vector<int32_t>{queries},true,true,aux_layers_);entry->prepare({warm},config_);std::cerr<<"[INFO ] ATB prefill 启动预热\n";entry->program.eager(runtime_.stream);check(aclrtSynchronizeStream(runtime_.stream),"ATB prefill 启动预热");cold_prefill_.emplace(std::vector<int32_t>{queries},std::move(entry));prefill_lru_.push_front({true,{queries}});auto& decode=standard_entry({warm},false);decode.prepare({warm},config_);decode.program.run(runtime_.stream);check(aclrtSynchronizeStream(runtime_.stream),"ATB decode 启动预热");for(auto* model:{target_.get(),draft_.get()})if(model)for(int l=0;l<model->net.layers;++l){zero(model->net.keys[l]);zero(model->net.values[l]);}std::cerr<<"[INFO ] ATB Qwen3 执行程序准备完成\n";
  }
  ~QwenAscendModel()override{runtime_.enter();aclrtSynchronizeStream(runtime_.stream);}
  ModelOutput run(const Batch& batch)override{
    runtime_.enter();for(auto& s:batch.sequences)active_.insert(s->id);if(batch.prefill||!draft_)return standard(batch.sequences,batch.prefill);
    auto fits=[&](const Seq& s){return s->tokens.size()+config_.spec_num_tokens+1<=size_t(config_.max_model_len);};
    if(!std::all_of(batch.sequences.begin(),batch.sequences.end(),fits)){
      // 临近窗口上限的请求单独收尾，不拖慢同批仍可运行完整投机轮的请求。
      std::vector<Seq> normal,tail;std::vector<size_t> ni,ti;ModelOutput out;out.tokens.resize(batch.sequences.size());for(size_t i=0;i<batch.sequences.size();++i)if(fits(batch.sequences[i])){normal.push_back(batch.sequences[i]);ni.push_back(i);}else{tail.push_back(batch.sequences[i]);ti.push_back(i);}
      if(!normal.empty()){auto value=run(Batch{false,normal});for(size_t i=0;i<ni.size();++i)out.tokens[ni[i]]=std::move(value.tokens[i]);out.rounds=std::move(value.rounds);out.samples=std::move(value.samples);}auto value=limited(tail);for(size_t i=0;i<ti.size();++i)out.tokens[ti[i]]=std::move(value.tokens[i]);return out;
    }
    int n=batch.sequences.size(),shape=bucket(n);auto& entry=spec_entries_[shape];if(!entry){entry=std::make_unique<DFlashEntry>(*target_,*draft_,config_,shape,aux_layers_);capture(entry->program);}entry->prepare(batch.sequences,config_);if(entry->program.graph){entry->program.run(runtime_.stream);++spec_;}else{entry->program.eager(runtime_.stream);++eager_;}auto host=read_ids(entry->output,runtime_.stream);verify_rows_+=n*entry->rows;ModelOutput out;int width=2*entry->k+2;
    for(int i=0;i<n;++i){auto base=host.begin()+i*width;std::vector<int64_t> proposed(base,base+entry->k),verified(base+entry->k,base+entry->k+entry->rows);int count=base[width-1];require(count>=1&&count<=entry->rows,"设备投机提交长度非法");std::vector<int64_t> committed(verified.begin(),verified.begin()+count);require(committed==accept_greedy(proposed,verified),"设备/主机投机接受结果不一致");out.tokens.push_back(committed);out.rounds.push_back({{"request_id",batch.sequences[i]->id},{"draft_ids",proposed},{"verify_ids",verified},{"accepted_count",count-1},{"committed_ids",committed},{"context_length",batch.sequences[i]->tokens.size()-1}});projected_+=count;}
    if(config_.trace_logits)for(int i=0;i<n;++i){out.samples.push_back(qwen_top_logits(entry->draft_logits,i*entry->k,entry->k,"draft",runtime_.stream));out.samples.push_back(qwen_top_logits(entry->verify_logits,i*entry->rows,entry->rows,"verify",runtime_.stream));}return out;
  }
  void drop(uint64_t id)override{active_.erase(id);}
  bool prefix_page_ready(const std::vector<int64_t>&,int32_t)const override{return true;}
  void evict_prefix_page(const std::vector<int64_t>&,int32_t)override{}
  Json snapshot()const override{return {{"backend","ascend310p"},{"attention","atb-flash-paged-splitfuse-nz"},{"maturity","candidate"},{"target_profile",target_->net.config.at("model_type")},{"spec_method",draft_?"dflash":"none"},{"quantized_compute",false},{"decode_graph_replays",decode_},{"draft_graph_replays",spec_},{"verify_graph_replays",spec_},{"spec_graph_replays",spec_},{"verify_rows",verify_rows_},{"prefill_tokens",prefill_},{"eager_rounds",eager_},{"graph_capture_failures",capture_failed_},{"draft_projected_tokens",projected_},{"active_requests",active_.size()},{"graph_buckets",fe::graph_buckets(config_.max_num_seqs)},{"prefill_plan_entries",prefill_lru_.size()},{"prefill_plan_evictions",prefill_evicted_},{"graph_execution",draft_?"batched-whole-dflash-round":"batched-decode"},{"device_spec_commit",bool(draft_)},{"device_auxiliary",bool(draft_)},{"qkv_packed",true},{"gate_up_packed",true},{"weights","fractal-nz"}};}
};
}
