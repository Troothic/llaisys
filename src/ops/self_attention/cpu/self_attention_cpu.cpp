#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <limits>

// 自注意力实现:
// 1. A = Q @ K^T * scale
// 2. 应用因果mask（上三角设为-inf）
// 3. A = softmax(A)
// 4. Y = A @ V
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t qlen, size_t kvlen, 
                     size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    // 计算每个head对应多少个kv head（GQA支持）
    size_t heads_per_kv = n_heads / n_kv_heads;
    
    // 为每个head计算注意力
    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / heads_per_kv;  // 对应的kv head索引
        
        // 对每个query位置
        for (size_t qi = 0; qi < qlen; qi++) {
            // 计算当前query在全局上下文中的位置（用于因果mask）
            // 假设query的位置从 (kvlen - qlen) 开始
            size_t q_pos = kvlen - qlen + qi;
            
            // 计算注意力分数并存储
            std::vector<float> attn_scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();
            
            // 计算 Q @ K^T * scale
            for (size_t ki = 0; ki < kvlen; ki++) {
                // 因果mask: 只看当前位置及之前的位置
                if (ki > q_pos) {
                    attn_scores[ki] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = qi * n_heads * head_dim + h * head_dim + d;
                    size_t k_idx = ki * n_kv_heads * head_dim + kv_h * head_dim + d;
                    
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = static_cast<float>(q[q_idx]);
                        k_val = static_cast<float>(k[k_idx]);
                    }
                    score += q_val * k_val;
                }
                score *= scale;
                attn_scores[ki] = score;
                if (score > max_score) max_score = score;
            }
            
            // Softmax: exp(x - max) / sum(exp(x - max))
            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (attn_scores[ki] > -std::numeric_limits<float>::infinity() * 0.5f) {
                    attn_scores[ki] = std::exp(attn_scores[ki] - max_score);
                    sum_exp += attn_scores[ki];
                } else {
                    attn_scores[ki] = 0.0f;
                }
            }
            for (size_t ki = 0; ki < kvlen; ki++) {
                attn_scores[ki] /= sum_exp;
            }
            
            // 计算 Attention @ V
            for (size_t d = 0; d < head_dim; d++) {
                float result = 0.0f;
                for (size_t vi = 0; vi < kvlen; vi++) {
                    size_t v_idx = vi * n_kv_heads * head_dim + kv_h * head_dim + d;
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = static_cast<float>(v[v_idx]);
                    }
                    result += attn_scores[vi] * v_val;
                }
                
                size_t out_idx = qi * n_heads * head_dim + h * head_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(result);
                } else {
                    attn_val[out_idx] = static_cast<T>(result);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
