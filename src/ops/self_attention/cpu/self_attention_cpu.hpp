#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// self_attention算子CPU实现
// attn_val: 输出张量 [qlen, n_heads, head_dim]
// q: 查询张量 [qlen, n_heads, head_dim]
// k: 键张量 [kvlen, n_kv_heads, head_dim]
// v: 值张量 [kvlen, n_kv_heads, head_dim]
// scale: 缩放因子 (通常是 1/sqrt(head_dim))
// type: 数据类型
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim);
}
