#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 验证输入
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index必须是I64类型");
    ASSERT(weight->ndim() == 2, "Embedding: weight必须是2D张量");
    ASSERT(index->ndim() == 1, "Embedding: index必须是1D张量");
    ASSERT(out->ndim() == 2, "Embedding: out必须是2D张量");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: 所有张量必须连续");
    
    size_t idx_len = index->shape()[0];
    size_t emb_dim = weight->shape()[1];
    
    // 目前只支持CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), idx_len, emb_dim);
    }
    
    core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                             out->dtype(), idx_len, emb_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops

