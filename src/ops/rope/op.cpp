#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 验证输入
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids必须是I64类型");
    
    ASSERT(out->ndim() == 3, "RoPE: out必须是3D张量 [seq_len, n_heads, head_dim]");
    ASSERT(in->ndim() == 3, "RoPE: in必须是3D张量 [seq_len, n_heads, head_dim]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids必须是1D张量");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: 所有张量必须连续");
    
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids长度必须等于seq_len");
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim必须是偶数");
    
    // 目前只支持CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        out->dtype(), seq_len, n_heads, head_dim);
    }
    
    core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        out->dtype(), seq_len, n_heads, head_dim);
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

