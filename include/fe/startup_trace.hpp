#pragma once
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace fe {
// 仅用于设备启动诊断；默认关闭，不改变 benchmark 的 JSON 输出。
inline bool startup_trace_enabled(){
  static const bool enabled=[] {const char* value=std::getenv("FLASHENGINE_TRACE_STARTUP");return value&&std::string(value)=="1";}();
  return enabled;
}
inline void startup_trace(const std::string& stage){
  if(!startup_trace_enabled())return;
  static const auto origin=std::chrono::steady_clock::now();
  const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-origin).count();
  std::cerr<<"[STARTUP +"<<elapsed<<"s] "<<stage<<std::endl;
}
}
