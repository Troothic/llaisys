from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.tensor import llaisysTensor_t

from pathlib import Path
from ctypes import Structure, POINTER, c_float, c_size_t, c_int64, c_int, c_void_p, byref
import numpy as np
import json


# C结构体定义
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),       # llaisysDataType_t
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


# 注册C函数
def _load_qwen2_api(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        c_int,                       # device
        POINTER(c_int),              # device_ids
        c_int,                       # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelResetKVCache.argtypes = [c_void_p]
    lib.llaisysQwen2ModelResetKVCache.restype = None


# 加载API
_load_qwen2_api(LIB_LLAISYS)


class Qwen2:
    # 权重名称映射
    WEIGHT_MAP = {
        "model.embed_tokens.weight": "in_embed",
        "lm_head.weight": "out_embed",
        "model.norm.weight": "out_norm_w",
    }

    LAYER_WEIGHT_MAP = {
        "input_layernorm.weight": "attn_norm_w",
        "self_attn.q_proj.weight": "attn_q_w",
        "self_attn.q_proj.bias": "attn_q_b",
        "self_attn.k_proj.weight": "attn_k_w",
        "self_attn.k_proj.bias": "attn_k_b",
        "self_attn.v_proj.weight": "attn_v_w",
        "self_attn.v_proj.bias": "attn_v_b",
        "self_attn.o_proj.weight": "attn_o_w",
        "post_attention_layernorm.weight": "mlp_norm_w",
        "mlp.gate_proj.weight": "mlp_gate_w",
        "mlp.up_proj.weight": "mlp_up_w",
        "mlp.down_proj.weight": "mlp_down_w",
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 加载配置
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # 创建元数据
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16.value  # 使用BF16
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config.get("num_key_value_heads", meta.nh)
        meta.dh = meta.hs // meta.nh
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 4096)
        meta.voc = config["vocab_size"]
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        
        self._meta = meta
        self._nlayer = meta.nlayer
        self._device = device
        
        print(f"Qwen2 Model Config:")
        print(f"  - layers: {meta.nlayer}, hidden: {meta.hs}, heads: {meta.nh}")
        print(f"  - vocab: {meta.voc}, intermediate: {meta.di}")
        
        # 创建模型
        device_id = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), device.value, device_id, 1
        )
        
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        
        # 获取权重指针
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        
        # 加载权重
        print("Loading weights...")
        self._load_weights(model_path)
        print("Weights loaded!")

    def _load_weights(self, model_path):
        """从safetensors文件加载权重"""
        try:
            from safetensors.torch import load_file
            for file in sorted(model_path.glob("*.safetensors")):
                print(f"  Loading {file.name}...")
                weights = load_file(str(file))
                for name, tensor in weights.items():
                    # 转换为numpy，处理bfloat16
                    if tensor.dtype == getattr(__import__('torch'), 'bfloat16'):
                        # 先转float32再处理
                        np_data = tensor.float().numpy()
                    else:
                        np_data = tensor.numpy()
                    self._set_weight(name, np_data)
        except ImportError:
            print("Error: safetensors or torch not found")
            raise

    def _set_weight(self, name: str, data: np.ndarray):
        """设置单个权重"""
        # 检查是否是全局权重
        if name in self.WEIGHT_MAP:
            attr_name = self.WEIGHT_MAP[name]
            tensor = getattr(self._weights.contents, attr_name)
            if tensor:
                self._load_tensor(tensor, data)
            return
        
        # 检查是否是层权重
        for layer_suffix, attr_name in self.LAYER_WEIGHT_MAP.items():
            if layer_suffix in name:
                # 提取层编号
                # 格式: model.layers.0.self_attn.q_proj.weight
                parts = name.split(".")
                try:
                    layer_idx = int(parts[2])
                except (IndexError, ValueError):
                    continue
                
                attr_array = getattr(self._weights.contents, attr_name)
                if attr_array and layer_idx < self._nlayer:
                    tensor = attr_array[layer_idx]
                    if tensor:
                        self._load_tensor(tensor, data)
                return


    def _load_tensor(self, tensor_ptr, data: np.ndarray):
        """将numpy数据加载到tensor"""
        if data.dtype == np.float32:
            # 转换为BF16: 取float32的高16位
            data_int = data.view(np.uint32)
            data_bf16 = ((data_int >> 16) & 0xFFFF).astype(np.uint16)
            data_bytes = data_bf16.tobytes()
        elif data.dtype == np.float16:
            data_bytes = data.tobytes()
        else:
            # 假设已经是BF16或其他格式
            data_bytes = data.tobytes()
        
        # 创建ctypes buffer并加载
        buffer = (c_int64 * (len(data_bytes) // 8 + 1))()
        import ctypes
        ctypes.memmove(buffer, data_bytes, len(data_bytes))
        LIB_LLAISYS.tensorLoad(tensor_ptr, ctypes.cast(buffer, c_void_p))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本"""
        # 重置KV缓存
        LIB_LLAISYS.llaisysQwen2ModelResetKVCache(self._model)
        
        # 转换输入为ctypes数组
        input_tokens = list(inputs)
        token_array = (c_int64 * len(input_tokens))(*input_tokens)
        
        # 首先处理所有输入token
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, token_array, len(input_tokens)
        )
        
        # 生成新token
        output_tokens = list(inputs)
        output_tokens.append(next_token)
        
        if max_new_tokens is None:
            max_new_tokens = 128
        
        for step in range(max_new_tokens - 1):
            # 检查是否结束
            if next_token == self._meta.end_token:
                break
            
            # 只传入最后一个token进行推理
            last_token = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, last_token, 1
            )
            
            output_tokens.append(next_token)
        
        return output_tokens

    def __del__(self):
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
