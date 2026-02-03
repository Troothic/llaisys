#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

// embedding实现模板：从权重矩阵中按索引复制行
template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t idx_len, size_t emb_dim) {
    for (size_t i = 0; i < idx_len; i++) {
        // 获取当前索引指向的权重行
        int64_t row_idx = index[i];
        const T *src_row = weight + row_idx * emb_dim;
        T *dst_row = out + i * emb_dim;
        
        // 复制整行
        std::memcpy(dst_row, src_row, emb_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t idx_len, size_t emb_dim) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            idx_ptr,
            reinterpret_cast<const float *>(weight),
            idx_len, emb_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            idx_ptr,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            idx_len, emb_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            idx_ptr,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            idx_len, emb_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
