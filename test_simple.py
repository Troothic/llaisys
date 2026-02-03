"""简单测试LLAISYS Qwen2模型加载和推理"""
import sys
sys.path.insert(0, "./python")

import llaisys
from llaisys.models import Qwen2
from llaisys.libllaisys import DeviceType

MODEL_PATH = r"F:\cursor_learn\Infinitensor\2025\llaisys\models\DeepSeek-R1-Distill-Qwen-1.5B"

print("=" * 50)
print("Testing LLAISYS Qwen2 Model")
print("=" * 50)

try:
    print("\n1. Creating model...")
    model = Qwen2(MODEL_PATH, DeviceType.CPU)
    print("   Model created successfully!")
    
    print("\n2. Testing inference with simple input...")
    # 使用简单的token ID测试
    test_input = [1, 2, 3, 4, 5]  # 简单token序列
    
    print(f"   Input tokens: {test_input}")
    output = model.generate(test_input, max_new_tokens=3)
    print(f"   Output tokens: {output}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
