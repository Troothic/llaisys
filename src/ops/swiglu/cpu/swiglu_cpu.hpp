#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// swiglu算子CPU实现
// out: 输出张量
// gate: 门控张量
// up: 上投影张量
// 计算: out = up * gate * sigmoid(gate) = up * swish(gate)
// type: 数据类型
// numel: 元素个数
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel);
}
