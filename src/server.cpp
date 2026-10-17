#define BOOST_ERROR_CODE_HEADER_ONLY
#include "fe/server.hpp"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <optional>
#include <sstream>

namespace fe {
namespace net=boost::asio;namespace beast=boost::beast;namespace http=beast::http;using tcp=net::ip::tcp;
namespace {
struct Context {Engine& engine;Tokenizer& tokenizer;const ServerConfig& config;int64_t vocab_size;};
struct Prepared {GenerateRequest request;bool chat=false,stream=false,usage=false;std::vector<std::string> stop;};
Prepared prepare(Context& ctx,const Json& body,bool chat){
  if(!body.is_object())throw InputError("请求体必须为 JSON 对象");
  const std::set<std::string> allowed{"model","prompt","messages","max_tokens","temperature","top_p","top_k","n","stream","stop","stream_options","chat_template_kwargs"};
  for(auto it=body.begin();it!=body.end();++it)if(!allowed.contains(it.key()))throw InputError("不支持的请求字段："+it.key());
  if(body.contains("model")&&body["model"]!=ctx.config.model)throw InputError("未知 model");
  if(!body.value("n",Json(1)).is_number_integer()||body.value("n",Json(1)).get<int>()!=1)throw InputError("仅支持 n=1");
  if(body.value("temperature",0.)!=0.)throw InputError("首版仅支持 greedy temperature=0");
  if(body.contains("max_tokens")&&!body["max_tokens"].is_number_integer())throw InputError("max_tokens 必须为整数");
  if(body.contains("top_k")&&!body["top_k"].is_number_integer())throw InputError("top_k 必须为整数");
  for(auto field:{"stream_options","chat_template_kwargs"})if(body.contains(field)&&!body[field].is_object())throw InputError(std::string(field)+" 必须为对象");
  for(auto field:{"stream_options","chat_template_kwargs"})if(body.contains(field))for(auto it=body[field].begin();it!=body[field].end();++it)if(it.key()!=(std::string(field)=="stream_options"?"include_usage":"enable_thinking"))throw InputError("不支持的嵌套配置："+it.key());
  double top_p=body.value("top_p",1.);int top_k=body.value("top_k",0);if(top_p<=0||top_p>1||top_k<0)throw InputError("非法 top_p/top_k");
  Prepared p;p.chat=chat;p.stream=body.value("stream",false);p.usage=body.value("stream_options",Json::object()).value("include_usage",false);
  if(body.contains("stop")){if(body["stop"].is_string())p.stop.push_back(body["stop"]);else p.stop=body["stop"].get<std::vector<std::string>>();for(auto& s:p.stop)if(s.empty())throw InputError("stop 不可为空");}
  int limit=body.value("max_tokens",128);if(limit<1)throw InputError("max_tokens 必须为正");
  auto& config=ctx.engine.config();int reserve=std::min(body.contains("max_tokens")?limit:ctx.config.reserve_output,int(config.max_model_len)-1);int cap=config.max_model_len-reserve;
  if(chat){
    auto messages=body.at("messages");if(!messages.is_array()||messages.empty())throw InputError("messages 不能为空");
    for(auto& m:messages){std::string role=m.at("role");if((role!="system"&&role!="user"&&role!="assistant")||!m.at("content").is_string())throw InputError("仅支持文本 system/user/assistant 消息");for(auto it=m.begin();it!=m.end();++it)if(it.key()!="role"&&it.key()!="content")throw InputError("不支持的消息字段："+it.key());}
    bool thinking=body.value("chat_template_kwargs",Json::object()).value("enable_thinking",ctx.config.thinking);
    auto render=[&](){return ctx.tokenizer.encode(ctx.tokenizer.render(messages,thinking));};p.request.token_ids=render();
    size_t system=0;while(system<messages.size()&&messages[system]["role"]=="system")++system;
    while(p.request.token_ids.size()>size_t(cap)&&messages.size()>system+1){messages.erase(messages.begin()+system);p.request.token_ids=render();}
    auto truncate_last=[&](){auto text_ids=ctx.tokenizer.encode(messages.back()["content"]);int64_t allow=text_ids.size();while(allow>=0){size_t keep=std::min(size_t(allow),text_ids.size());messages.back()["content"]=ctx.tokenizer.decode({text_ids.end()-keep,text_ids.end()});p.request.token_ids=render();if(p.request.token_ids.size()<=size_t(cap))return;allow-=std::max(int64_t(1),int64_t(p.request.token_ids.size())-cap);text_ids=ctx.tokenizer.encode(messages.back()["content"]);}};
    if(p.request.token_ids.size()>size_t(cap)&&messages.size()>system)truncate_last();
    if(p.request.token_ids.size()>size_t(cap)){auto last=messages.back();messages=Json::array({last});truncate_last();}
    if(p.request.token_ids.size()>size_t(cap))throw InputError("上下文无法容纳聊天模板");
  }else{auto prompt=body.at("prompt");if(prompt.is_string())p.request.token_ids=ctx.tokenizer.encode(prompt);else p.request.token_ids=prompt.get<std::vector<int64_t>>();if(p.request.token_ids.size()>size_t(cap))p.request.token_ids.erase(p.request.token_ids.begin(),p.request.token_ids.end()-cap);}
  if(p.request.token_ids.empty())throw InputError("prompt 不能为空");for(auto id:p.request.token_ids)if(id<0||id>=ctx.vocab_size)throw InputError("token ID 越界");
  p.request.max_tokens=std::min(limit,int(config.max_model_len-p.request.token_ids.size()));return p;
}
class Session:public std::enable_shared_from_this<Session>{
  beast::tcp_stream stream_;beast::flat_buffer buffer_;http::request_parser<http::string_body> parser_;Context& ctx_;
  RequestHandle handle_;Prepared prepared_;std::vector<int64_t> tokens_;std::deque<std::shared_ptr<std::string>> writes_;size_t bytes_=0,emitted_=0,prompt_tokens_=0;
  std::atomic<size_t> mailbox_bytes_{0};std::atomic<bool> mailbox_overflow_{false};
  bool writing_=false,finishing_=false,closed_=false;int64_t created_=std::time(nullptr);std::string request_id_;
  void close(){if(closed_)return;closed_=true;handle_.cancel();beast::error_code ec;stream_.socket().cancel(ec);stream_.socket().shutdown(tcp::socket::shutdown_both,ec);stream_.socket().close(ec);}
  void response(http::status status,const std::string& body,const std::string& type="application/json"){
    auto message=std::make_shared<http::response<http::string_body>>(status,11);message->set(http::field::content_type,type);message->set(http::field::server,"FlashEngine/0.1.0");message->keep_alive(false);message->body()=body;message->prepare_payload();stream_.expires_after(std::chrono::seconds(30));http::async_write(stream_,*message,[self=shared_from_this(),message](beast::error_code,size_t){self->close();});
  }
  void queue(std::string data){if(closed_)return;bytes_+=data.size();if(bytes_>256*1024){close();return;}writes_.push_back(std::make_shared<std::string>(std::move(data)));write();}
  void write(){if(writing_||closed_)return;if(writes_.empty()){if(finishing_)close();return;}writing_=true;stream_.expires_after(std::chrono::seconds(30));auto item=writes_.front();net::async_write(stream_,net::buffer(*item),[self=shared_from_this(),item](beast::error_code ec,size_t){if(ec){self->close();return;}self->bytes_-=item->size();self->writes_.pop_front();self->writing_=false;self->write();});}
  Json envelope(Json choices,bool chunk=false)const{return {{"id",request_id_},{"object",prepared_.chat?(chunk?"chat.completion.chunk":"chat.completion"):"text_completion"},{"created",created_},{"model",ctx_.config.model},{"choices",std::move(choices)}};}
  void chunk(Json choices){queue("data: "+envelope(std::move(choices),true).dump()+"\n\n");}
  void event(OutputEvent event){
    if(closed_||finishing_)return;tokens_.insert(tokens_.end(),event.token_ids.begin(),event.token_ids.end());auto text=ctx_.tokenizer.decode(tokens_);bool finished=event.finished;std::string reason=event.finish_reason;
    size_t stop_at=std::string::npos;for(auto& stop:prepared_.stop)stop_at=std::min(stop_at,text.find(stop));if(stop_at!=std::string::npos){text.resize(stop_at);finished=true;reason="stop";handle_.cancel();}
    size_t safe=text.size();if(!finished){while(safe>=3&&text.compare(safe-3,3,"\xef\xbf\xbd")==0)safe-=3;size_t hold=0;for(auto& stop:prepared_.stop)for(size_t n=1;n<stop.size()&&n<=safe;++n)if(text.compare(safe-n,n,stop,0,n)==0)hold=std::max(hold,n);safe-=hold;}
    if(safe<emitted_){close();return;}std::string delta=text.substr(emitted_,safe-emitted_);emitted_=safe;
    if(prepared_.stream){if(!delta.empty()){Json choice={{"index",0},{"finish_reason",nullptr}};choice[prepared_.chat?"delta":"text"]=prepared_.chat?Json{{"content",delta}}:Json(delta);chunk(Json::array({choice}));}
      if(finished){Json choice={{"index",0},{"finish_reason",reason}};choice[prepared_.chat?"delta":"text"]=prepared_.chat?Json::object():Json("");chunk(Json::array({choice}));if(prepared_.usage){auto final=envelope(Json::array(),true);final["usage"]={{"prompt_tokens",prompt_tokens_},{"completion_tokens",tokens_.size()},{"total_tokens",prompt_tokens_+tokens_.size()}};queue("data: "+final.dump()+"\n\n");}queue("data: [DONE]\n\n");finishing_=true;write();}}
    else if(finished){Json choice={{"index",0},{"finish_reason",reason}};choice[prepared_.chat?"message":"text"]=prepared_.chat?Json{{"role","assistant"},{"content",text}}:Json(text);auto result=envelope(Json::array({choice}));result["usage"]={{"prompt_tokens",prompt_tokens_},{"completion_tokens",tokens_.size()},{"total_tokens",prompt_tokens_+tokens_.size()}};finishing_=true;response(http::status::ok,result.dump());}
  }
  void disconnect_watch(){stream_.socket().async_wait(tcp::socket::wait_read,[self=shared_from_this()](beast::error_code ec){if(ec||self->closed_)return;char c;beast::error_code error;size_t got=self->stream_.socket().receive(net::buffer(&c,1),tcp::socket::message_peek,error);if(error||!got){self->close();return;}self->close();});}
  void route(http::request<http::string_body> request){
    std::string path(request.target());try{
      if(path.rfind("/v1/",0)==0&&!ctx_.config.api_key.empty()&&request[http::field::authorization]!="Bearer "+ctx_.config.api_key){response(http::status::unauthorized,Json{{"error",{{"message","需要有效 Bearer key"},{"type","authentication_error"}}}}.dump());return;}
      if(request.method()==http::verb::get){
        if(path=="/health"){response(http::status::ok,"{\"status\":\"ok\"}");return;}
        if(path=="/v1/models"){response(http::status::ok,Json{{"object","list"},{"data",Json::array({{{"id",ctx_.config.model},{"object","model"},{"created",created_},{"owned_by","flashengine"}}})}}.dump());return;}
        if(path=="/v1/stats"){response(http::status::ok,ctx_.engine.runtime_snapshot().dump());return;}
        if(path=="/metrics"||path=="/v1/metrics"){auto snapshot=ctx_.engine.runtime_snapshot();std::ostringstream text;for(auto it=snapshot["metrics"].begin();it!=snapshot["metrics"].end();++it)text<<it.key()<<' '<<it.value()<<'\n';response(http::status::ok,text.str(),"text/plain; version=0.0.4");return;}
      }
      if(request.method()!=http::verb::post||(path!="/v1/chat/completions"&&path!="/v1/completions")){response(http::status::not_found,"{\"error\":{\"message\":\"unknown endpoint\"}}");return;}
      prepared_=prepare(ctx_,parse_json(request.body()),path=="/v1/chat/completions");prompt_tokens_=prepared_.request.token_ids.size();
      // 图外驱动到网络 strand 的邮箱也设限，防止尚未进入发送队列的 post 无限堆积。
      std::weak_ptr<Session> weak=shared_from_this();handle_=ctx_.engine.submit(std::move(prepared_.request),[weak](OutputEvent event){if(auto self=weak.lock()){
        if(self->mailbox_overflow_.load())return;size_t cost=sizeof(OutputEvent)+event.token_ids.size()*sizeof(int64_t)+event.finish_reason.size();
        if(self->mailbox_bytes_.fetch_add(cost)+cost>64*1024){self->mailbox_bytes_.fetch_sub(cost);if(!self->mailbox_overflow_.exchange(true))net::post(self->stream_.get_executor(),[self]{self->close();});return;}
        net::post(self->stream_.get_executor(),[self,cost,event=std::move(event)]()mutable{self->mailbox_bytes_.fetch_sub(cost);self->event(std::move(event));});
      }});request_id_="fe-"+std::to_string(handle_.request_id);
      if(prepared_.stream){queue("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n");if(prepared_.chat)chunk(Json::array({{{"index",0},{"delta",{{"role","assistant"},{"content",""}}},{"finish_reason",nullptr}}}));}
      disconnect_watch();
    }catch(const std::length_error& error){response(http::status::too_many_requests,Json{{"error",{{"message",error.what()},{"type","overloaded"}}}}.dump());}
    catch(const std::exception& error){response(http::status::bad_request,Json{{"error",{{"message",error.what()},{"type","invalid_request_error"}}}}.dump());}
  }
public:
  Session(tcp::socket socket,Context& ctx):stream_(std::move(socket)),ctx_(ctx){parser_.body_limit(1<<20);}
  void start(){stream_.expires_after(std::chrono::seconds(30));http::async_read(stream_,buffer_,parser_,[self=shared_from_this()](beast::error_code ec,size_t){if(ec){self->close();return;}self->route(self->parser_.release());});}
};
}
int serve(Engine& engine,Tokenizer& tokenizer,const ServerConfig& config,int64_t vocab_size){
  net::io_context io(2);Context context{engine,tokenizer,config,vocab_size};tcp::acceptor acceptor(net::make_strand(io),{net::ip::make_address(config.host),config.port});
  std::function<void()> accept;accept=[&]{acceptor.async_accept(net::make_strand(io),[&](beast::error_code ec,tcp::socket socket){if(!ec&&socket.is_open()){socket.set_option(tcp::no_delay(true),ec);if(!ec)std::make_shared<Session>(std::move(socket),context)->start();}if(acceptor.is_open())accept();});};accept();
  // accept/rearm 与信号关闭必须共用 strand；acceptor 不是可并发操作的共享对象。
  net::signal_set signals(acceptor.get_executor(),SIGINT,SIGTERM);signals.async_wait([&](beast::error_code,int){beast::error_code ignored;acceptor.close(ignored);io.stop();});
  // 任一事件循环异常都先停止并 join；禁止在栈展开时销毁 joinable thread。
  std::exception_ptr failure;std::mutex failure_mutex;auto run=[&]{try{io.run();}catch(...){std::lock_guard lock(failure_mutex);if(!failure)failure=std::current_exception();io.stop();}};
  std::cerr<<"[INFO ] Listening http://"<<config.host<<':'<<config.port<<" model="<<config.model<<'\n';std::thread worker(run);run();worker.join();engine.stop();if(failure)std::rethrow_exception(failure);return 0;
}
}
