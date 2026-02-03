#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
// argmax算子CPU实现
// max_idx: 存储最大值索引
// max_val: 存储最大值
// vals: 输入张量数据
// type: 数据类型
// numel: 元素个数
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
}
