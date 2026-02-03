#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RoPE实现: 对每个位置应用旋转位置编码
// phi = pos / theta^(2j/d)
// a' = a*cos(phi) - b*sin(phi)
// b' = b*cos(phi) + a*sin(phi)
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                // 计算旋转角度: phi = pos / theta^(2j/d)
                float freq_exp = static_cast<float>(2 * j) / static_cast<float>(head_dim);
                float freq = std::pow(theta, freq_exp);
                float phi = static_cast<float>(pos) / freq;
                
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // 获取输入的a和b分量
                size_t idx_a = s * n_heads * head_dim + h * head_dim + j;
                size_t idx_b = s * n_heads * head_dim + h * head_dim + half_dim + j;
                
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[idx_a]);
                    b = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a = static_cast<float>(in[idx_a]);
                    b = static_cast<float>(in[idx_b]);
                }
                
                // 应用旋转
                float a_prime = a * cos_phi - b * sin_phi;
                float b_prime = b * cos_phi + a * sin_phi;
                
                // 写入输出
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(a_prime);
                    out[idx_b] = llaisys::utils::cast<T>(b_prime);
                } else {
                    out[idx_a] = static_cast<T>(a_prime);
                    out[idx_b] = static_cast<T>(b_prime);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ptr, theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
