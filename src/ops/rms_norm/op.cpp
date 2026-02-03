#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 验证输入
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    ASSERT(out->ndim() == 2, "RMSNorm: out必须是2D张量");
    ASSERT(in->ndim() == 2, "RMSNorm: in必须是2D张量");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight必须是1D张量");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: 所有张量必须连续");
    
    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];
    
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(weight->shape()[0] == cols, "RMSNorm: weight长度必须等于列数");
    
    // 目前只支持CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps,
                            out->dtype(), rows, cols);
    }
    
    core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps,
                            out->dtype(), rows, cols);
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

