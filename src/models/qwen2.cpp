#include "qwen2.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>


namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id) {
    allocate_layer_weights();
    init_kv_cache();
}

void Qwen2Model::allocate_layer_weights() {
    size_t nlayer = _meta.nlayer;
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    size_t di = _meta.di;
    size_t voc = _meta.voc;
    
    // 全局权重
    _weights.in_embed = Tensor::create({voc, hs}, _meta.dtype, _device_type, _device_id);
    _weights.out_embed = Tensor::create({voc, hs}, _meta.dtype, _device_type, _device_id);
    _weights.out_norm_w = Tensor::create({hs}, _meta.dtype, _device_type, _device_id);
    
    // 层权重
    _weights.attn_norm_w.resize(nlayer);
    _weights.attn_q_w.resize(nlayer);
    _weights.attn_q_b.resize(nlayer);
    _weights.attn_k_w.resize(nlayer);
    _weights.attn_k_b.resize(nlayer);
    _weights.attn_v_w.resize(nlayer);
    _weights.attn_v_b.resize(nlayer);
    _weights.attn_o_w.resize(nlayer);
    _weights.mlp_norm_w.resize(nlayer);
    _weights.mlp_gate_w.resize(nlayer);
    _weights.mlp_up_w.resize(nlayer);
    _weights.mlp_down_w.resize(nlayer);
    
    for (size_t i = 0; i < nlayer; i++) {
        _weights.attn_norm_w[i] = Tensor::create({hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_q_w[i] = Tensor::create({nh * dh, hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_q_b[i] = Tensor::create({nh * dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_k_w[i] = Tensor::create({nkvh * dh, hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_k_b[i] = Tensor::create({nkvh * dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_v_w[i] = Tensor::create({nkvh * dh, hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_v_b[i] = Tensor::create({nkvh * dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_o_w[i] = Tensor::create({hs, nh * dh}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_norm_w[i] = Tensor::create({hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_gate_w[i] = Tensor::create({di, hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_up_w[i] = Tensor::create({di, hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_down_w[i] = Tensor::create({hs, di}, _meta.dtype, _device_type, _device_id);
    }
}

void Qwen2Model::init_kv_cache() {
    size_t nlayer = _meta.nlayer;
    _kv_cache.k_cache.resize(nlayer);
    _kv_cache.v_cache.resize(nlayer);
    
    // 预分配KV缓存空间 [maxseq, nkvh, dh]
    std::vector<size_t> cache_shape = {_meta.maxseq, _meta.nkvh, _meta.dh};
    for (size_t i = 0; i < nlayer; i++) {
        _kv_cache.k_cache[i] = Tensor::create(cache_shape, _meta.dtype, _device_type, _device_id);
        _kv_cache.v_cache[i] = Tensor::create(cache_shape, _meta.dtype, _device_type, _device_id);
    }
    _kv_cache.seq_len = 0;
}


void Qwen2Model::reset_kv_cache() {
    _kv_cache.seq_len = 0;
}

// 注意力层前向传播
void Qwen2Model::forward_attention(size_t layer, tensor_t hidden, tensor_t pos_ids) {
    size_t seq_len = hidden->shape()[0];
    size_t hs = _meta.hs; (void)hs;  // suppress unused warning
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    
    // 1. 计算Q, K, V
    // Q: [seq_len, nh*dh]
    std::vector<size_t> q_shape = {seq_len, nh * dh};
    auto q = Tensor::create(q_shape, _meta.dtype, _device_type, _device_id);
    ops::linear(q, hidden, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
    
    // K: [seq_len, nkvh*dh]
    std::vector<size_t> kv_shape = {seq_len, nkvh * dh};
    auto k = Tensor::create(kv_shape, _meta.dtype, _device_type, _device_id);
    ops::linear(k, hidden, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
    
    // V: [seq_len, nkvh*dh]
    auto v = Tensor::create(kv_shape, _meta.dtype, _device_type, _device_id);
    ops::linear(v, hidden, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
    
    // 2. Reshape为多头格式
    // Q: [seq_len, nh, dh]
    auto q_heads = q->view({seq_len, nh, dh});
    // K, V: [seq_len, nkvh, dh]
    auto k_heads = k->view({seq_len, nkvh, dh});
    auto v_heads = v->view({seq_len, nkvh, dh});
    
    // 3. 应用RoPE
    auto q_rope = Tensor::create({seq_len, nh, dh}, _meta.dtype, _device_type, _device_id);
    auto k_rope = Tensor::create({seq_len, nkvh, dh}, _meta.dtype, _device_type, _device_id);
    ops::rope(q_rope, q_heads, pos_ids, _meta.theta);
    ops::rope(k_rope, k_heads, pos_ids, _meta.theta);
    
    // 4. 更新KV缓存
    size_t kv_start = _kv_cache.seq_len;
    // 将新的K, V写入缓存
    // 简化处理：直接复制到缓存的对应位置
    auto k_cache = _kv_cache.k_cache[layer];
    auto v_cache = _kv_cache.v_cache[layer];
    
    // 复制数据到KV缓存的[kv_start:kv_start+seq_len]位置
    size_t elem_size = k_rope->elementSize();
    size_t copy_bytes = seq_len * nkvh * dh * elem_size;
    std::byte* k_dst = k_cache->data() + kv_start * nkvh * dh * elem_size;
    std::byte* v_dst = v_cache->data() + kv_start * nkvh * dh * elem_size;
    std::memcpy(k_dst, k_rope->data(), copy_bytes);
    std::memcpy(v_dst, v_heads->data(), copy_bytes);
    
    // 5. 计算注意力
    size_t total_kv_len = kv_start + seq_len;
    
    // 获取完整的KV缓存视图 [total_kv_len, nkvh, dh]
    auto k_full = k_cache->slice(0, 0, total_kv_len);
    auto v_full = v_cache->slice(0, 0, total_kv_len);
    
    // 计算缩放因子
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    
    // 计算注意力 [seq_len, nh, dh]
    auto attn_out = Tensor::create({seq_len, nh, dh}, _meta.dtype, _device_type, _device_id);
    ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
    
    // 6. 输出投影
    // Reshape到 [seq_len, nh*dh]
    auto attn_flat = attn_out->view({seq_len, nh * dh});
    
    // O投影: [seq_len, hs]
    // O投影结果直接写回hidden (不做残差连接，由调用方处理)
    ops::linear(hidden, attn_flat, _weights.attn_o_w[layer], nullptr);
}

// MLP层前向传播
void Qwen2Model::forward_mlp(size_t layer, tensor_t hidden) {
    size_t seq_len = hidden->shape()[0];
    size_t hs = _meta.hs; (void)hs;  // suppress unused warning
    size_t di = _meta.di;
    
    // 1. gate投影
    std::vector<size_t> mlp_shape = {seq_len, di};
    auto gate = Tensor::create(mlp_shape, _meta.dtype, _device_type, _device_id);
    ops::linear(gate, hidden, _weights.mlp_gate_w[layer], nullptr);
    
    // 2. up投影
    auto up = Tensor::create(mlp_shape, _meta.dtype, _device_type, _device_id);
    ops::linear(up, hidden, _weights.mlp_up_w[layer], nullptr);
    
    // 3. SwiGLU激活
    auto mlp_hidden = Tensor::create(mlp_shape, _meta.dtype, _device_type, _device_id);
    ops::swiglu(mlp_hidden, gate, up);
    
    // 4. down投影结果直接写回hidden (不做残差连接，由调用方处理)
    ops::linear(hidden, mlp_hidden, _weights.mlp_down_w[layer], nullptr);
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    size_t hs = _meta.hs;
    size_t nlayer = _meta.nlayer;
    
    // 1. 创建位置ID张量
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    std::vector<int64_t> pos_data(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_data[i] = static_cast<int64_t>(_kv_cache.seq_len + i);
    }
    pos_ids->load(reinterpret_cast<std::byte*>(pos_data.data()));
    
    // 2. 创建token ID张量
    auto token_tensor = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    token_tensor->load(reinterpret_cast<std::byte*>(const_cast<int64_t*>(token_ids)));
    
    // 3. 嵌入查找: [ntoken, hs]
    auto hidden = Tensor::create({ntoken, hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(hidden, token_tensor, _weights.in_embed);
    
    // 4. 通过所有Transformer层
    for (size_t layer = 0; layer < nlayer; layer++) {
        // === Attention Block ===
        // residual = hidden
        auto residual = Tensor::create({ntoken, hs}, _meta.dtype, _device_type, _device_id);
        std::memcpy(residual->data(), hidden->data(), ntoken * hs * hidden->elementSize());
        
        // hidden = rms_norm(hidden)
        auto normed = Tensor::create({ntoken, hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(normed, hidden, _weights.attn_norm_w[layer], _meta.epsilon);
        
        // hidden = attention(normed) (forward_attention会修改normed并加残差到自身)
        forward_attention(layer, normed, pos_ids);
        
        // hidden = attention_output + residual
        ops::add(hidden, normed, residual);
        
        // === MLP Block ===
        // residual = hidden
        std::memcpy(residual->data(), hidden->data(), ntoken * hs * hidden->elementSize());
        
        // mlp_input = rms_norm(hidden)
        auto mlp_normed = Tensor::create({ntoken, hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(mlp_normed, hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        
        // mlp_output = mlp(mlp_normed)
        forward_mlp(layer, mlp_normed);
        
        // hidden = mlp_output + residual
        ops::add(hidden, mlp_normed, residual);
    }

    
    // 5. 最终LayerNorm
    auto final_hidden = Tensor::create({ntoken, hs}, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(final_hidden, hidden, _weights.out_norm_w, _meta.epsilon);
    
    // 6. 只取最后一个位置的hidden state
    // final_hidden: [ntoken, hs], 取最后一行
    auto last_hidden = final_hidden->slice(0, ntoken - 1, ntoken);  // [1, hs]
    
    // 7. 计算logits (使用out_embed作为lm_head)
    // logits = last_hidden @ out_embed.T
    // out_embed: [voc, hs], 需要转置
    auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, last_hidden, _weights.out_embed, nullptr);
    
    // 8. 找到最大值的索引 (argmax)
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    
    // 将logits展平为1D
    auto logits_flat = logits->view({_meta.voc});
    ops::argmax(max_idx, max_val, logits_flat);
    
    // 9. 更新KV缓存长度
    _kv_cache.seq_len += ntoken;
    
    // 10. 返回预测的token
    int64_t next_token;
    std::memcpy(&next_token, max_idx->data(), sizeof(int64_t));
    return next_token;
}

} // namespace llaisys::models
