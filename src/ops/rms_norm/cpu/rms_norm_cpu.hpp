#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// rms_norm算子CPU实现
// out: 输出张量 [rows, cols]
// in: 输入张量 [rows, cols]
// weight: 权重向量 [cols]
// eps: 防止除零的小值
// type: 数据类型
// rows, cols: 矩阵维度
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t rows, size_t cols);
}
