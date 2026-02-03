#include "../models/qwen2.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include <cstring>
#include <vector>


// C API实现 - 导出给Python使用

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model* model;
    std::vector<LlaisysTensor*> tensor_wrappers;  // 存储所有的张量包装器
    LlaisysQwen2Weights weights_c;  // C接口权重结构
};

// 辅助函数：创建LlaisysTensor包装器
static LlaisysTensor* wrap_tensor(llaisys::tensor_t t, std::vector<LlaisysTensor*>& wrappers) {
    if (!t) return nullptr;
    auto* wrapper = new LlaisysTensor();
    wrapper->tensor = t;
    wrappers.push_back(wrapper);
    return wrapper;
}

extern "C" {

// 创建模型
__export LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice
) {
    // 转换Meta
    llaisys::models::Qwen2Meta cpp_meta;
    cpp_meta.dtype = meta->dtype;
    cpp_meta.nlayer = meta->nlayer;
    cpp_meta.hs = meta->hs;
    cpp_meta.nh = meta->nh;
    cpp_meta.nkvh = meta->nkvh;
    cpp_meta.dh = meta->dh;
    cpp_meta.di = meta->di;
    cpp_meta.maxseq = meta->maxseq;
    cpp_meta.voc = meta->voc;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.theta = meta->theta;
    cpp_meta.end_token = meta->end_token;
    
    int device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    
    auto* wrapper = new LlaisysQwen2Model();
    wrapper->model = new llaisys::models::Qwen2Model(cpp_meta, device, device_id);
    
    // 初始化C接口权重结构的指针
    size_t nlayer = meta->nlayer;
    wrapper->weights_c.attn_norm_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_q_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_q_b = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_k_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_k_b = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_v_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_v_b = new llaisysTensor_t[nlayer];
    wrapper->weights_c.attn_o_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.mlp_norm_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.mlp_gate_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.mlp_up_w = new llaisysTensor_t[nlayer];
    wrapper->weights_c.mlp_down_w = new llaisysTensor_t[nlayer];
    
    // 初始化为nullptr
    for (size_t i = 0; i < nlayer; i++) {
        wrapper->weights_c.attn_norm_w[i] = nullptr;
        wrapper->weights_c.attn_q_w[i] = nullptr;
        wrapper->weights_c.attn_q_b[i] = nullptr;
        wrapper->weights_c.attn_k_w[i] = nullptr;
        wrapper->weights_c.attn_k_b[i] = nullptr;
        wrapper->weights_c.attn_v_w[i] = nullptr;
        wrapper->weights_c.attn_v_b[i] = nullptr;
        wrapper->weights_c.attn_o_w[i] = nullptr;
        wrapper->weights_c.mlp_norm_w[i] = nullptr;
        wrapper->weights_c.mlp_gate_w[i] = nullptr;
        wrapper->weights_c.mlp_up_w[i] = nullptr;
        wrapper->weights_c.mlp_down_w[i] = nullptr;
    }
    
    return wrapper;
}

// 销毁模型
__export void llaisysQwen2ModelDestroy(LlaisysQwen2Model* model) {
    if (model) {
        // 清理所有张量包装器
        for (auto* w : model->tensor_wrappers) {
            delete w;
        }
        
        delete model->model;
        delete[] model->weights_c.attn_norm_w;
        delete[] model->weights_c.attn_q_w;
        delete[] model->weights_c.attn_q_b;
        delete[] model->weights_c.attn_k_w;
        delete[] model->weights_c.attn_k_b;
        delete[] model->weights_c.attn_v_w;
        delete[] model->weights_c.attn_v_b;
        delete[] model->weights_c.attn_o_w;
        delete[] model->weights_c.mlp_norm_w;
        delete[] model->weights_c.mlp_gate_w;
        delete[] model->weights_c.mlp_up_w;
        delete[] model->weights_c.mlp_down_w;
        delete model;
    }
}

// 获取权重指针
__export LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model* model) {
    if (!model) return nullptr;
    
    auto* cpp_weights = model->model->weights();
    auto* c_weights = &model->weights_c;
    auto& wrappers = model->tensor_wrappers;
    
    // 同步指针 - 将C++的tensor_t包装为LlaisysTensor
    c_weights->in_embed = wrap_tensor(cpp_weights->in_embed, wrappers);
    c_weights->out_embed = wrap_tensor(cpp_weights->out_embed, wrappers);
    c_weights->out_norm_w = wrap_tensor(cpp_weights->out_norm_w, wrappers);
    
    size_t nlayer = model->model->meta().nlayer;
    for (size_t i = 0; i < nlayer; i++) {
        c_weights->attn_norm_w[i] = wrap_tensor(cpp_weights->attn_norm_w[i], wrappers);
        c_weights->attn_q_w[i] = wrap_tensor(cpp_weights->attn_q_w[i], wrappers);
        c_weights->attn_q_b[i] = wrap_tensor(cpp_weights->attn_q_b[i], wrappers);
        c_weights->attn_k_w[i] = wrap_tensor(cpp_weights->attn_k_w[i], wrappers);
        c_weights->attn_k_b[i] = wrap_tensor(cpp_weights->attn_k_b[i], wrappers);
        c_weights->attn_v_w[i] = wrap_tensor(cpp_weights->attn_v_w[i], wrappers);
        c_weights->attn_v_b[i] = wrap_tensor(cpp_weights->attn_v_b[i], wrappers);
        c_weights->attn_o_w[i] = wrap_tensor(cpp_weights->attn_o_w[i], wrappers);
        c_weights->mlp_norm_w[i] = wrap_tensor(cpp_weights->mlp_norm_w[i], wrappers);
        c_weights->mlp_gate_w[i] = wrap_tensor(cpp_weights->mlp_gate_w[i], wrappers);
        c_weights->mlp_up_w[i] = wrap_tensor(cpp_weights->mlp_up_w[i], wrappers);
        c_weights->mlp_down_w[i] = wrap_tensor(cpp_weights->mlp_down_w[i], wrappers);
    }
    
    return c_weights;
}

// 推理
__export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    if (!model) return -1;
    return model->model->infer(token_ids, ntoken);
}

// 重置KV缓存
__export void llaisysQwen2ModelResetKVCache(LlaisysQwen2Model* model) {
    if (model) {
        model->model->reset_kv_cache();
    }
}

} // extern "C"
