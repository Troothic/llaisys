#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 检查张量是否在内存中连续存储
    // 从最后一维开始检查：strides[n-1]应该是1
    // 对于每一维i，strides[i]应该等于strides[i+1] * shape[i+1]
    if (_meta.shape.empty()) {
        return true;  // 空张量视为连续
    }
    
    size_t ndim = _meta.shape.size();
    ptrdiff_t expected_stride = 1;
    
    // 从最后一维向前检查
    for (size_t i = ndim; i > 0; --i) {
        size_t dim = i - 1;
        if (_meta.strides[dim] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[dim]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查order的有效性
    size_t ndim = _meta.shape.size();
    ASSERT(order.size() == ndim, "Permute: order维度与张量维度不匹配");
    
    // 验证order是否是有效的排列（0到ndim-1的排列）
    std::vector<bool> seen(ndim, false);
    for (size_t i = 0; i < ndim; ++i) {
        ASSERT(order[i] < ndim, "Permute: order索引越界");
        ASSERT(!seen[order[i]], "Permute: order中有重复的索引");
        seen[order[i]] = true;
    }
    
    // 根据order重新排列shape和strides
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (size_t s : shape) {
        new_numel *= s;
    }
    
    // 检查元素总数是否匹配
    ASSERT(new_numel == this->numel(), "View: 新形状的元素总数与原张量不匹配");
    
    // 检查张量是否连续（只有连续张量才能做view）
    ASSERT(this->isContiguous(), "View: 张量必须连续才能进行view操作");
    
    // 计算新的strides（从后往前计算，最后一维stride为1）
    size_t ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim);
    ptrdiff_t stride = 1;
    for (size_t i = ndim; i > 0; --i) {
        new_strides[i - 1] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i - 1]);
    }
    
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 验证参数
    ASSERT(dim < _meta.shape.size(), "Slice: 维度索引越界");
    ASSERT(start < end, "Slice: start必须小于end");
    ASSERT(end <= _meta.shape[dim], "Slice: end越界");
    
    // 计算新的offset（字节为单位）
    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();
    
    // 复制shape并更新切片维度的大小
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;
    
    // strides保持不变
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 获取当前设备的运行时API
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    // 计算需要复制的字节数
    size_t bytes = this->numel() * this->elementSize();
    
    // 确定复制方向
    llaisysMemcpyKind_t kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;  // 主机到主机
    } else {
        kind = LLAISYS_MEMCPY_H2D;  // 主机到设备
    }
    
    // 执行内存复制
    core::context().runtime().api()->memcpy_sync(
        this->data(),               // 目标地址
        src_,                       // 源地址
        bytes,                      // 字节数
        kind                        // 复制方向
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
