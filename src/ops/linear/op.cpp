#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 验证输入
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    
    ASSERT(out->ndim() == 2, "Linear: out必须是2D张量");
    ASSERT(in->ndim() == 2, "Linear: in必须是2D张量");
    ASSERT(weight->ndim() == 2, "Linear: weight必须是2D张量");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: 所有张量必须连续");
    if (bias) ASSERT(bias->isContiguous(), "Linear: bias必须连续");
    
    // out: [M, N], in: [M, K], weight: [N, K]
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "Linear: 输出形状不匹配");
    ASSERT(weight->shape()[1] == K, "Linear: weight的K维度不匹配");
    
    std::byte *bias_data = bias ? bias->data() : nullptr;
    
    // 目前只支持CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                          out->dtype(), M, N, K);
    }
    
    core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                          out->dtype(), M, N, K);
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

