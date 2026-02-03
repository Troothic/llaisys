#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
// embedding算子CPU实现
// out: 输出张量 [idx_len, emb_dim]
// index: 索引张量 [idx_len], I64类型
// weight: 权重矩阵 [vocab_size, emb_dim]
// type: 数据类型
// idx_len: 索引长度
// emb_dim: 嵌入维度
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t idx_len, size_t emb_dim);
}
