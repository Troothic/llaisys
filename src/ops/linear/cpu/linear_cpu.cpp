#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// linear实现模板: out[m][n] = sum_k(in[m][k] * weight[n][k]) + bias[n]
// 注意：weight是[N, K]形状，计算时相当于 X @ W^T
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            // 计算 in[m, :] @ weight[n, :]^T = sum_k(in[m,k] * weight[n,k])
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float in_val, w_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    in_val = llaisys::utils::cast<float>(in[m * K + k]);
                    w_val = llaisys::utils::cast<float>(weight[n * K + k]);
                } else {
                    in_val = static_cast<float>(in[m * K + k]);
                    w_val = static_cast<float>(weight[n * K + k]);
                }
                sum += in_val * w_val;
            }
            
            // 加上偏置
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[n]);
                } else {
                    sum += static_cast<float>(bias[n]);
                }
            }
            
            // 写入输出
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[m * N + n] = llaisys::utils::cast<T>(sum);
            } else {
                out[m * N + n] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias ? reinterpret_cast<const float *>(bias) : nullptr,
            M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
            M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
            M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
