#include "fe/tokenizer.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <minja/minja.hpp>
#include <array>
#include <queue>
#include <set>
#include <unordered_map>

namespace fe {
using namespace std::string_literals;
namespace {
nlohmann::json read_vocab_json(const fs::path& path){
  // 大词表不能用 ordered_json 的线性键查找，否则解析为 O(vocab²)，ASan 下尤甚。
  std::vector<std::set<std::string>> keys;
  return nlohmann::json::parse(read_text(path),[&](int,nlohmann::json::parse_event_t event,nlohmann::json& value){
    if(event==nlohmann::json::parse_event_t::object_start)keys.emplace_back();
    if(event==nlohmann::json::parse_event_t::key)require(keys.back().insert(value.get<std::string>()).second,"tokenizer JSON 重复键");
    if(event==nlohmann::json::parse_event_t::object_end)keys.pop_back();
    return true;
  });
}
std::string utf8(unsigned cp){std::string s;if(cp<128)s+=char(cp);else if(cp<2048){s+=char(192|(cp>>6));s+=char(128|(cp&63));}else if(cp<65536){s+=char(224|(cp>>12));s+=char(128|((cp>>6)&63));s+=char(128|(cp&63));}else{ s+=char(240|(cp>>18));s+=char(128|((cp>>12)&63));s+=char(128|((cp>>6)&63));s+=char(128|(cp&63));}return s;}
std::string unicode_text(const std::string& input,bool nfc,bool replace=false){
  require(input.size()<=INT32_MAX,"文本过长");UErrorCode error=U_ZERO_ERROR;int32_t length=0;
  if(replace)u_strFromUTF8WithSub(nullptr,0,&length,input.data(),input.size(),0xfffd,nullptr,&error);
  else u_strFromUTF8(nullptr,0,&length,input.data(),input.size(),&error);
  require(error==U_BUFFER_OVERFLOW_ERROR||U_SUCCESS(error),"非法 UTF-8");error=U_ZERO_ERROR;
  std::vector<UChar> chars(length+1);
  if(replace)u_strFromUTF8WithSub(chars.data(),chars.size(),&length,input.data(),input.size(),0xfffd,nullptr,&error);
  else u_strFromUTF8(chars.data(),chars.size(),&length,input.data(),input.size(),&error);
  require(U_SUCCESS(error),"UTF-8 转换失败");
  if(nfc){const auto* normalizer=unorm2_getNFCInstance(&error);require(U_SUCCESS(error),"NFC 初始化失败");error=U_ZERO_ERROR;int count=unorm2_normalize(normalizer,chars.data(),length,nullptr,0,&error);require(error==U_BUFFER_OVERFLOW_ERROR||U_SUCCESS(error),"NFC 长度失败");error=U_ZERO_ERROR;std::vector<UChar> norm(count+1);length=unorm2_normalize(normalizer,chars.data(),length,norm.data(),norm.size(),&error);require(U_SUCCESS(error),"NFC 归一化失败");chars=std::move(norm);}
  error=U_ZERO_ERROR;int32_t bytes=0;u_strToUTF8(nullptr,0,&bytes,chars.data(),length,&error);require(error==U_BUFFER_OVERFLOW_ERROR||U_SUCCESS(error),"UTF-8 长度失败");error=U_ZERO_ERROR;std::string result(bytes,'\0');u_strToUTF8(result.data(),bytes,&bytes,chars.data(),length,&error);require(U_SUCCESS(error),"UTF-8 输出失败");return result;
}
}
struct Tokenizer::Impl {
  std::unordered_map<std::string,int64_t> vocab,rank;
  std::unordered_map<int64_t,std::string> inverse;
  std::vector<std::pair<std::string,int64_t>> added;
  std::set<int64_t> special;
  std::array<std::string,256> byte_encoder;
  std::unordered_map<std::string,char> byte_decoder;
  std::shared_ptr<pcre2_code> regex;
  std::shared_ptr<minja::TemplateNode> tmpl;
  bool nfc=false;

  void bpe(const std::string& piece,std::vector<int64_t>& result)const {
    if(piece.empty())return;
    struct Node{std::string text;int prev,next,version=0;bool alive=true;};
    struct Item{int64_t rank;int left,right,lv,rv;bool operator>(const Item& b)const{return rank!=b.rank?rank>b.rank:left>b.left;}};
    std::vector<Node> nodes;nodes.reserve(piece.size());
    for(size_t i=0;i<piece.size();++i)nodes.push_back({byte_encoder[uint8_t(piece[i])],int(i)-1,i+1==piece.size()?-1:int(i)+1});
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
    auto add=[&](int left){if(left<0||!nodes[left].alive)return;int right=nodes[left].next;if(right<0)return;auto it=rank.find(nodes[left].text+"\0"s+nodes[right].text);if(it!=rank.end())queue.push({it->second,left,right,nodes[left].version,nodes[right].version});};
    for(size_t i=0;i<nodes.size();++i)add(i);
    while(!queue.empty()){
      auto entry=queue.top();queue.pop();auto& a=nodes[entry.left];auto& b=nodes[entry.right];
      if(!a.alive||!b.alive||a.version!=entry.lv||b.version!=entry.rv||a.next!=entry.right)continue;
      a.text+=b.text;a.next=b.next;++a.version;b.alive=false;++b.version;if(b.next>=0)nodes[b.next].prev=entry.left;
      add(a.prev);add(entry.left);
    }
    for(int idx=0;idx>=0;idx=nodes[idx].next){auto it=vocab.find(nodes[idx].text);require(it!=vocab.end(),"BPE 结果未在词表中");result.push_back(it->second);}
  }
  void regular(const std::string& text,std::vector<int64_t>& ids)const {
    if(text.empty())return;
    auto input=unicode_text(text,nfc);
    std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)> match(pcre2_match_data_create_from_pattern(regex.get(),nullptr),pcre2_match_data_free);
    size_t offset=0;
    while(offset<input.size()){
      int rc=pcre2_match(regex.get(),reinterpret_cast<PCRE2_SPTR>(input.data()),input.size(),offset,0,match.get(),nullptr);
      require(rc>=0,"Qwen 预分词失败");auto* span=pcre2_get_ovector_pointer(match.get());require(span[0]==offset&&span[1]>offset,"预分词未覆盖输入");bpe(input.substr(offset,span[1]-offset),ids);offset=span[1];
    }
  }
};
Tokenizer::Tokenizer(const fs::path& root):impl_(std::make_shared<Impl>()){
  using namespace std::string_literals;
  auto data=read_vocab_json(root/"tokenizer/tokenizer.json");auto& p=*impl_;
  require(data["model"]["type"]=="BPE","不支持的 tokenizer model");
  require(data["normalizer"].is_null()||data["normalizer"]==nlohmann::json{{"type","NFC"}},"不支持的 normalizer");p.nfc=!data["normalizer"].is_null();
  auto pre=data.at("pre_tokenizer");require(pre["type"]=="Sequence"&&pre["pretokenizers"].size()==2,"不支持的预分词 pipeline");
  auto split=pre["pretokenizers"][0],bytes=pre["pretokenizers"][1];
  require(split["type"]=="Split"&&split["behavior"]=="Isolated"&&!split["invert"].get<bool>()&&bytes["type"]=="ByteLevel"&&!bytes["add_prefix_space"].get<bool>()&&!bytes["use_regex"].get<bool>(),"不支持的 Qwen 预分词配置");
  std::string pattern=split["pattern"]["Regex"];int error;PCRE2_SIZE where;
  auto* regex=pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),pattern.size(),PCRE2_UTF|PCRE2_UCP,&error,&where,nullptr);
  require(regex,"Unicode 正则编译失败");p.regex={regex,pcre2_code_free};
  for(auto it=data["model"]["vocab"].begin();it!=data["model"]["vocab"].end();++it){int64_t id=it.value();require(id>=0&&p.inverse.emplace(id,it.key()).second,"词表 ID 重复");p.vocab.emplace(it.key(),id);}
  int64_t rank=0;for(auto& merge:data["model"]["merges"]){std::string left,right;if(merge.is_array()){left=merge[0];right=merge[1];}else{std::string pair=merge;auto space=pair.find(' ');require(space!=std::string::npos,"非法 merge");left=pair.substr(0,space);right=pair.substr(space+1);}require(p.rank.emplace(left+"\0"s+right,rank++).second,"重复 BPE merge");}
  for(auto& item:data["added_tokens"]){require(!item["single_word"].get<bool>()&&!item["lstrip"].get<bool>()&&!item["rstrip"].get<bool>()&&!item["normalized"].get<bool>(),"不支持的 added token 规则");std::string token=item["content"];int64_t id=item["id"];require(!token.empty(),"空 added token");p.added.emplace_back(token,id);p.inverse[id]=token;if(item["special"].get<bool>())p.special.insert(id);}
  int extra=0;for(int b=0;b<256;++b){bool visible=(b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174);auto token=utf8(visible?b:256+extra++);p.byte_encoder[b]=token;p.byte_decoder[token]=char(b);}
  p.tmpl=minja::Parser::parse(read_text(root/"tokenizer/chat_template.jinja"),minja::Options{true,true,false});
}
std::vector<int64_t> Tokenizer::encode(const std::string& text)const {
  std::vector<int64_t> ids;size_t offset=0;
  while(offset<text.size()){
    size_t start=text.size(),length=0;int64_t id=-1;
    for(const auto& [token,value]:impl_->added){auto found=text.find(token,offset);if(found<start||(found==start&&token.size()>length)){start=found;length=token.size();id=value;}}
    impl_->regular(text.substr(offset,start-offset),ids);if(id<0||start==std::string::npos)break;ids.push_back(id);offset=start+length;
  }
  return ids;
}
std::string Tokenizer::decode(const std::vector<int64_t>& ids,bool skip_special)const {
  std::string out;for(auto id:ids){auto it=impl_->inverse.find(id);require(it!=impl_->inverse.end(),"token ID 越界");if(skip_special&&impl_->special.contains(id))continue;
    bool added=false;for(auto& [token,value]:impl_->added)if(value==id){out+=token;added=true;break;}if(added)continue;
    const auto& token=it->second;for(size_t i=0;i<token.size();){unsigned char b=token[i];size_t n=b<128?1:b<224?2:b<240?3:4;require(i+n<=token.size(),"非法词表 UTF-8");auto found=impl_->byte_decoder.find(token.substr(i,n));require(found!=impl_->byte_decoder.end(),"非 byte-level 词表字符");out+=found->second;i+=n;}
  }
  return unicode_text(out,false,true);
}
std::string Tokenizer::render(const Json& messages,bool thinking)const {
  require(messages.is_array()&&!messages.empty(),"messages 必须是非空数组");
  auto context=minja::Context::make(minja::Value(Json{{"messages",messages},{"tools",nullptr},{"add_generation_prompt",true},{"enable_thinking",thinking},{"bos_token",""},{"eos_token","<|im_end|>"}}));
  return impl_->tmpl->render(context);
}
void Tokenizer::verify_golden(const Json& golden)const {
  for(auto& item:golden.at("texts")){auto ids=encode(item.at("text"));require(ids==item.at("token_ids").get<std::vector<int64_t>>(),"tokenizer encode golden 不一致");require(decode(ids,false)==item.at("decoded"),"tokenizer decode golden 不一致");}
  for(auto& item:golden.at("chats")){auto text=render(item.at("messages"),item.at("enable_thinking"));require(text==item.at("rendered"),"chat template 字节不一致");require(encode(text)==item.at("token_ids").get<std::vector<int64_t>>(),"chat template token 不一致");}
}
}
