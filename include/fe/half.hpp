#pragma once
#include <bit>
#include <cstdint>

namespace fe {
// 主机侧 IEEE-754 round-to-nearest-even；避免依赖任一厂商 half 类型。
inline uint16_t float_to_half(float value) {
  uint32_t bits=std::bit_cast<uint32_t>(value),sign=(bits>>16)&0x8000,m=bits&0x7fffff;
  int exponent=int((bits>>23)&255),e=exponent-127+15;
  if(exponent==255)return uint16_t(sign|0x7c00|(m?0x200:0));
  if(e>=31)return uint16_t(sign|0x7c00);
  if(e<=0){
    if(e<-10)return uint16_t(sign);
    m|=0x800000;int shift=14-e;
    return uint16_t(sign|((m+((1u<<(shift-1))-1)+((m>>shift)&1))>>shift));
  }
  uint32_t rounded=m+0xfff+((m>>13)&1);
  return uint16_t(sign+(e<<10)+(rounded>>13));
}
}
