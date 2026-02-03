#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
// rope算子CPU实现 (旋转位置编码)
// out: 输出张量 [seq_len, n_heads, head_dim]
// in: 输入张量 [seq_len, n_heads, head_dim]
// pos_ids: 位置ID [seq_len], I64类型
// theta: 频率基数
// type: 数据类型
// seq_len, n_heads, head_dim: 张量维度
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim);
}
