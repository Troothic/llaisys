#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// linear算子CPU实现: Y = X * W^T + bias
// out: 输出张量 [M, N]
// in: 输入张量 [M, K]
// weight: 权重矩阵 [N, K] (注意：未转置)
// bias: 偏置向量 [N] (可为nullptr)
// type: 数据类型
// M, N, K: 矩阵维度
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K);
}
