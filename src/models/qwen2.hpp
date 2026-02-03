#pragma once

#include "../core/llaisys_core.hpp"
#include "../tensor/tensor.hpp"

#include <vector>
#include <memory>

namespace llaisys::models {

// Qwen2模型元数据
struct Qwen2Meta {
    llaisysDataType_t dtype;    // 数据类型 (通常是BF16)
    size_t nlayer;              // Transformer层数
    size_t hs;                  // 隐藏层大小 (hidden_size)
    size_t nh;                  // 注意力头数 (num_attention_heads)
    size_t nkvh;                // KV头数 (num_key_value_heads)
    size_t dh;                  // 每个头的维度 (head_dim = hs / nh)
    size_t di;                  // FFN中间层大小 (intermediate_size)
    size_t maxseq;              // 最大序列长度
    size_t voc;                 // 词表大小
    float epsilon;              // LayerNorm的epsilon
    float theta;                // RoPE的theta
    int64_t end_token;          // 结束token ID
};

// Qwen2模型权重
struct Qwen2Weights {
    tensor_t in_embed;          // [voc, hs] 输入嵌入
    tensor_t out_embed;         // [voc, hs] 输出嵌入(lm_head)
    tensor_t out_norm_w;        // [hs] 最终LayerNorm权重
    
    // 每层的权重 (数组大小=nlayer)
    std::vector<tensor_t> attn_norm_w;   // [hs] attention前的LayerNorm
    std::vector<tensor_t> attn_q_w;      // [nh*dh, hs] Q权重
    std::vector<tensor_t> attn_q_b;      // [nh*dh] Q偏置
    std::vector<tensor_t> attn_k_w;      // [nkvh*dh, hs] K权重
    std::vector<tensor_t> attn_k_b;      // [nkvh*dh] K偏置
    std::vector<tensor_t> attn_v_w;      // [nkvh*dh, hs] V权重
    std::vector<tensor_t> attn_v_b;      // [nkvh*dh] V偏置
    std::vector<tensor_t> attn_o_w;      // [hs, nh*dh] O权重
    std::vector<tensor_t> mlp_norm_w;    // [hs] MLP前的LayerNorm
    std::vector<tensor_t> mlp_gate_w;    // [di, hs] gate权重
    std::vector<tensor_t> mlp_up_w;      // [di, hs] up权重
    std::vector<tensor_t> mlp_down_w;    // [hs, di] down权重
};

// KV缓存
struct KVCache {
    std::vector<tensor_t> k_cache;  // [nlayer][maxseq, nkvh, dh]
    std::vector<tensor_t> v_cache;  // [nlayer][maxseq, nkvh, dh]
    size_t seq_len;                 // 当前缓存的序列长度
    
    KVCache() : seq_len(0) {}
};

// Qwen2模型
class Qwen2Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    KVCache _kv_cache;
    llaisysDeviceType_t _device_type;
    int _device_id;
    
    // 中间张量（复用以避免频繁分配）
    tensor_t _hidden;       // [seq_len, hs]
    tensor_t _residual;     // [seq_len, hs]
    tensor_t _attn_out;     // [seq_len, hs]
    tensor_t _q, _k, _v;    // Q/K/V张量
    tensor_t _attn_val;     // 注意力输出
    tensor_t _gate, _up;    // MLP中间结果
    
    // 前向传播辅助函数
    void forward_attention(size_t layer, tensor_t hidden, tensor_t pos_ids);
    void forward_mlp(size_t layer, tensor_t hidden);
    
public:
    Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;
    
    // 获取权重指针（用于加载权重）
    Qwen2Weights* weights() { return &_weights; }
    const Qwen2Meta& meta() const { return _meta; }
    
    // 初始化KV缓存
    void init_kv_cache();
    
    // 清空KV缓存
    void reset_kv_cache();
    
    // 单步推理：输入token_ids，返回下一个token
    int64_t infer(const int64_t* token_ids, size_t ntoken);
    
    // 分配层权重
    void allocate_layer_weights();
};

} // namespace llaisys::models
