#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RMS归一化实现: out[i] = weight[i] * in[i] / sqrt(mean(in^2) + eps)
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; row++) {
        // 计算当前行的平方和
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[row * cols + col]);
            } else {
                val = static_cast<float>(in[row * cols + col]);
            }
            sum_sq += val * val;
        }
        
        // 计算 1/sqrt(mean + eps)
        float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        
        // 应用归一化和权重
        for (size_t col = 0; col < cols; col++) {
            float in_val, w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                in_val = llaisys::utils::cast<float>(in[row * cols + col]);
                w_val = llaisys::utils::cast<float>(weight[col]);
            } else {
                in_val = static_cast<float>(in[row * cols + col]);
                w_val = static_cast<float>(weight[col]);
            }
            
            float result = w_val * in_val * rms_inv;
            
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[row * cols + col] = llaisys::utils::cast<T>(result);
            } else {
                out[row * cols + col] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t rows, size_t cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, rows, cols);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps, rows, cols);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps, rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
