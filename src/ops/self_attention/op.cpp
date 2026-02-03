#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 验证输入
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    ASSERT(attn_val->ndim() == 3, "SelfAttn: attn_val必须是3D张量 [qlen, n_heads, head_dim]");
    ASSERT(q->ndim() == 3, "SelfAttn: q必须是3D张量 [qlen, n_heads, head_dim]");
    ASSERT(k->ndim() == 3, "SelfAttn: k必须是3D张量 [kvlen, n_kv_heads, head_dim]");
    ASSERT(v->ndim() == 3, "SelfAttn: v必须是3D张量 [kvlen, n_kv_heads, head_dim]");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttn: 所有张量必须连续");
    
    size_t qlen = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    
    CHECK_SAME_SHAPE(attn_val->shape(), q->shape());
    ASSERT(k->shape()[2] == head_dim, "SelfAttn: k的head_dim必须与q相同");
    ASSERT(v->shape()[0] == kvlen, "SelfAttn: v的kvlen必须与k相同");
    ASSERT(v->shape()[1] == n_kv_heads, "SelfAttn: v的n_kv_heads必须与k相同");
    ASSERT(n_heads % n_kv_heads == 0, "SelfAttn: n_heads必须是n_kv_heads的倍数");
    
    // 目前只支持CPU
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                   attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim);
    }
    
    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                   attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim);
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

