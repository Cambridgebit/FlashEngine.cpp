#include "fe/engine.hpp"
#include <condition_variable>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace {
class FakeModel final:public fe::DeviceModel {
  bool fatal_;
public:
  explicit FakeModel(bool fatal=false):fatal_(fatal){}
  fe::ModelOutput run(const fe::Batch& batch)override{
    if(fatal_)throw std::runtime_error("测试注入的设备异常");
    fe::ModelOutput out;for(auto& s:batch.sequences)out.tokens.push_back(batch.prefill&&s->cached+s->scheduled<s->tokens.size()?std::vector<int64_t>{}:std::vector<int64_t>{7});return out;
  }
  void drop(uint64_t)override{}
  fe::Json snapshot()const override{return {{"backend","test-only"}};}
};
fe::Config config(){fe::Config c;c.max_model_len=64;c.max_num_seqs=2;c.max_num_batched_tokens=3;c.kvcache_block_size=16;c.num_kvcache_blocks=4;return c;}
}
int main(){try{
  std::mutex mutex;std::condition_variable cv;int done=0;size_t tokens=0;
  {
    fe::Engine engine(config(),std::make_unique<FakeModel>());
    auto sink=[&](fe::OutputEvent e){std::lock_guard lock(mutex);tokens+=e.token_ids.size();if(e.finished){++done;cv.notify_all();}};
    engine.submit({{1,2,3,4,5},4},sink);engine.submit({{2},3},sink);
    std::unique_lock lock(mutex);fe::require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return done==2;}),"引擎未完成并发请求");fe::require(tokens==7,"增量 token 丢失或重复");lock.unlock();
    bool invalid=false;try{engine.submit({{},2},sink);}catch(const fe::InputError&){invalid=true;}fe::require(invalid,"非法请求未局部拒绝");
  }
  {
    fe::Engine engine(config(),std::make_unique<FakeModel>());auto handle=engine.submit({{1},100},[](fe::OutputEvent){});handle.cancel();handle.cancel();engine.stop();engine.stop();
  }
  {
    auto c=config();c.enable_prefix_cache=true;done=0;
    fe::Engine engine(c,std::make_unique<FakeModel>());
    auto sink=[&](fe::OutputEvent e){if(e.finished){std::lock_guard lock(mutex);++done;cv.notify_all();}};
    engine.submit({std::vector<int64_t>(33,2),1},sink);
    {std::unique_lock lock(mutex);fe::require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return done==1;}),"冷前缀请求未完成");}
    engine.submit({std::vector<int64_t>(33,2),1},sink);
    {std::unique_lock lock(mutex);fe::require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return done==2;}),"热前缀请求未完成");}
    engine.stop();auto snapshot=engine.runtime_snapshot();auto metrics=snapshot.at("metrics");
    fe::require(metrics.at("flashengine_prefix_hit_requests_total")==1&&metrics.at("flashengine_prefix_cached_blocks_total")==2&&metrics.at("flashengine_kv_used_blocks")==0,"前缀快照/页引用计数错误");
  }
  // 引擎异常必须跨进程验证，以免测试进程被正常策略终止。
  pid_t child=fork();fe::require(child>=0,"fork 失败");
  if(child==0){fe::Engine engine(config(),std::make_unique<FakeModel>(true));engine.submit({{1},2},[](fe::OutputEvent){});std::this_thread::sleep_for(std::chrono::seconds(3));std::_Exit(99);}
  int status=0;waitpid(child,&status,0);fe::require(WIFEXITED(status)&&WEXITSTATUS(status)==70,"设备异常未 fail-fast");
  std::cout<<"engine concurrency/cancel/fail-fast PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
