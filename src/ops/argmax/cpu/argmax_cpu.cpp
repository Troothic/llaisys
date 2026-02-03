#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// argmax实现模板：遍历找最大值及其索引
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;
    
    T max = vals[0];
    int64_t idx = 0;
    
    for (size_t i = 1; i < numel; i++) {
        // 使用cast转换处理bf16和fp16类型的比较
        float curr, curr_max;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            curr = llaisys::utils::cast<float>(vals[i]);
            curr_max = llaisys::utils::cast<float>(max);
        } else {
            curr = static_cast<float>(vals[i]);
            curr_max = static_cast<float>(max);
        }
        
        if (curr > curr_max) {
            max = vals[i];
            idx = static_cast<int64_t>(i);
        }
    }
    
    *max_idx = idx;
    *max_val = max;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
