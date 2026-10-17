#include "fe/bundle.hpp"
#include "fe/profiles.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <limits>
#include <set>

namespace fe {
MappedFile::MappedFile(const fs::path& path) {
  int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW);require(fd>=0,"无法打开权重："+path.string());
  struct stat st{};if(fstat(fd,&st)!=0||!S_ISREG(st.st_mode)||st.st_size<=0){::close(fd);throw std::runtime_error("非法映射文件");}
  size_=st.st_size;data_=mmap(nullptr,size_,PROT_READ,MAP_PRIVATE,fd,0);::close(fd);
  require(data_!=MAP_FAILED,"mmap 失败");
}
MappedFile::~MappedFile(){if(data_ && data_!=MAP_FAILED)munmap(data_,size_);}
std::map<std::string,WeightView> parse_safetensors(std::shared_ptr<MappedFile> file) {
  require(file->size()>=8,"safetensors 截断");uint64_t length=0;
  for(int i=0;i<8;++i)length|=uint64_t(static_cast<unsigned char>(file->data()[i]))<<(8*i);
  require(length<=100000000 && length<=file->size()-8,"safetensors header 越界");
  require(length>0&&file->data()[8]=='{',"safetensors header 必须以对象起始字符开头");auto header=parse_json(std::string(file->data()+8,length));require(header.is_object(),"header 必须为对象");
  std::map<std::string,WeightView> result;std::vector<std::pair<uint64_t,uint64_t>> spans;
  const std::map<std::string,size_t> sizes{{"F16",2},{"BF16",2},{"F32",4},{"I64",8},{"BOOL",1}};
  for(auto it=header.begin();it!=header.end();++it){
    if(it.key()=="__metadata__"){require(it.value().is_object(),"safetensors metadata 必须为对象");for(auto& value:it.value())require(value.is_string(),"safetensors metadata 必须为字符串字典");continue;}
    auto& item=it.value();std::string dtype=item.at("dtype");
    require(sizes.contains(dtype),"不支持的 safetensors dtype");
    uint64_t count=1;std::vector<int64_t> shape;
    require(item.at("shape").is_array()&&item.at("shape").size()<=8,"非法维数");
    for(auto& dim:item.at("shape")){require(dim.is_number_integer()&&dim.get<int64_t>()>=0,"非法 shape");auto d=dim.get<int64_t>();require(!d||count<=uint64_t(std::numeric_limits<int64_t>::max())/d,"shape 溢出");count*=d;shape.push_back(d);}
    require(count<=std::numeric_limits<uint64_t>::max()/sizes.at(dtype),"字节数溢出");
    auto offsets=item.at("data_offsets");require(offsets.size()==2,"offset 数量错误");
    for(auto& value:offsets)require(value.is_number_integer()&&value.get<int64_t>()>=0,"非法 offset");
    uint64_t start=offsets[0],end=offsets[1];
    require(start<=end&&end<=file->size()-8-length&&end-start==count*sizes.at(dtype),"张量边界/shape 不一致");
    spans.emplace_back(start,end);result.emplace(it.key(),WeightView{file,size_t(start+8+length),size_t(end-start),dtype,shape});
  }
  std::sort(spans.begin(),spans.end());uint64_t cursor=0;
  for(auto [begin,end]:spans){require(begin==cursor,"safetensors 数据空洞或重叠");cursor=end;}
  require(cursor==file->size()-8-length,"未索引的 safetensors 数据");return result;
}
fs::path Bundle::path(const std::string& relative)const {
  fs::path rel(relative);require(!rel.empty()&&!rel.is_absolute(),"bundle 只接受相对路径");
  fs::path current=root;for(const auto& part:rel){require(part!=".."&&part!=".","bundle 路径穿越");current/=part;require(!fs::is_symlink(fs::symlink_status(current)),"bundle 禁止符号链接");}
  require(fs::is_regular_file(current),"bundle 缺失文件："+relative);return current;
}
Bundle::Bundle(const fs::path& directory,bool verify_hashes):root(fs::canonical(directory)) {
  manifest=read_json(path("bundle.json"));
  require((manifest.at("format_version")=="1.0.0"||manifest.at("format_version")=="1.1.0")&&compatible_contract(manifest.at("contract_version")),"不兼容的 bundle/contract 版本");
  for(auto it=manifest.at("files").begin();it!=manifest.at("files").end();++it){auto file=path(it.key());require(fs::file_size(file)==it.value().at("size").get<uint64_t>(),"bundle 文件大小不匹配："+it.key());if(verify_hashes)require(sha256(file)==it.value().at("sha256"),"bundle 哈希不匹配："+it.key());}
  for(auto name:{"engine/config.json","engine/weights.plan.json","tokenizer/tokenizer.json","tokenizer/runtime.json","tokenizer/chat_template.jinja","golden/tokenizer_chat.json"})require(manifest["files"].contains(name),"manifest 缺少必要文件");
  config=read_json(path("engine/config.json"));tokenizer_runtime=read_json(path("tokenizer/runtime.json"));validate_weights();
}
void Bundle::validate_weights(){
  auto plans=read_json(path("engine/weights.plan.json"));
  for(std::string kind:{"target","draft"}){
    if(!plans.contains(kind)){require(kind=="draft"&&config.at("draft").is_null(),"缺少权重装载计划");continue;}
    auto& destination=kind=="target"?target_weights:draft_weights;
    std::map<std::string,std::map<std::string,WeightView>> files;std::set<std::pair<std::string,std::string>> consumed;
    for(auto& entry:plans.at(kind).at("entries")){
      std::string filename=entry.at("file"),source=entry.at("source"),name=entry.at("destination");
      require(manifest.at("files").contains(filename),"权重未列入 manifest");
      if(!files.contains(filename))files[filename]=parse_safetensors(std::make_shared<MappedFile>(path(filename)));
      require(files.at(filename).contains(source),"源张量不存在");auto weight=files.at(filename).at(source);
      require(consumed.emplace(filename,source).second&&destination.emplace(name,weight).second,"权重重复装载");
      require(weight.offset==entry.at("offset").get<uint64_t>()&&weight.bytes==entry.at("bytes").get<uint64_t>()&&weight.shape==entry.at("shape").get<std::vector<int64_t>>()&&weight.dtype==entry.at("dtype"),"装载计划与 safetensors 不一致");
    }
    for(auto& entry:plans.at(kind).at("ignored")){
      std::string filename="weights/"+kind+"/"+entry.at("file").get<std::string>(),source=entry.at("source");
      if(!files.contains(filename)){require(manifest.at("files").contains(filename),"忽略张量未列入 manifest");files[filename]=parse_safetensors(std::make_shared<MappedFile>(path(filename)));}
      require(files.contains(filename)&&files.at(filename).contains(source)&&consumed.emplace(filename,source).second,"非法忽略权重");
    }
    for(auto& [filename,weights]:files)for(auto& [name,w]:weights)require(consumed.contains({filename,name}),"未消费的权重："+name);
    auto mc=config.at(kind);require(mc.is_object(),"模型配置缺失");
    int64_t h=mc.at("hidden_size"),vocab=mc.at("vocab_size");auto expected=expected_model_weights(mc,kind=="draft");
    if(kind=="draft"){
      auto ids=mc.at("target_layer_ids").get<std::vector<int>>();require(!ids.empty(),"草稿缺少 aux 层");
      std::set<int> seen;for(auto id:ids)require(id>=0&&id+1<config["target"]["num_hidden_layers"].get<int>()&&seen.insert(id).second,"草稿 aux 层越界或重复");
      require(h==config["target"]["hidden_size"].get<int64_t>()&&vocab==config["target"]["vocab_size"].get<int64_t>(),"草稿与目标维度不匹配");
      int64_t k=mc.at("num_tokens"),block=mc.at("block_size"),mask=mc.at("mask_token_id");
      require(k>0&&k<block&&mask>=0&&mask<vocab,"草稿 k/block/mask 配置非法");
    }
    for(auto& [name,shape]:expected)require(destination.contains(name)&&destination.at(name).shape==shape,"缺失或错误形状的权重："+name);
    for(auto& [name,weight]:destination){require(weight.dtype=="F16"||weight.dtype=="BF16"||weight.dtype=="F32","非浮点模型权重");require(expected.contains(name)||((name=="embed_tokens.weight"||name=="lm_head.weight")&&weight.shape==std::vector<int64_t>{vocab,h}),"未知目标权重："+name);}
  }
}
}
