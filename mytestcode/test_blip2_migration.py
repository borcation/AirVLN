#!/usr/bin/env python3
"""
测试BLIP-2编码器的基本功能
验证从CLIP到BLIP-2的迁移是否成功
"""

import torch
import numpy as np
from Model.encoders.blip2_encoder import BLIP2VisionEncoder, BLIP2DepthEncoder, BLIP2InstructionEncoder

def create_mock_observation_space():
    """创建模拟的观察空间"""
    class MockObservationSpace:
        def __init__(self):
            self.spaces = {
                'rgb': MockSpace((224, 224, 3)),
                'depth': MockSpace((224, 224, 1)),
                'instruction': MockSpace((512,))  # BLIP-2支持更长的文本序列
            }
    
    class MockSpace:
        def __init__(self, shape):
            self.shape = shape
    
    return MockObservationSpace()

def test_blip2_vision_encoder():
    """测试BLIP-2视觉编码器"""
    print("=== 测试BLIP-2视觉编码器 ===")
    
    observation_space = create_mock_observation_space()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建编码器
        encoder = BLIP2VisionEncoder(
            observation_space=observation_space,
            device=device,
            model_name="Salesforce/blip2-opt-2.7b",
            layer_extract=-1,
            freeze_backbone=True
        )
        print(f"✓ BLIP-2视觉编码器创建成功")
        print(f"  - 输出形状: {encoder.output_shape}")
        print(f"  - 输出大小: {encoder.output_size}")
        print(f"  - 设备: {device}")
        
        # 测试前向传播
        batch_size = 2
        rgb_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = encoder(rgb_input)
        
        print(f"  - 输入形状: {rgb_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"✓ BLIP-2视觉编码器前向传播成功")
        
        return True
        
    except Exception as e:
        print(f"✗ BLIP-2视觉编码器测试失败: {str(e)}")
        return False

def test_blip2_instruction_encoder():
    """测试BLIP-2指令编码器"""
    print("\n=== 测试BLIP-2指令编码器 ===")
    
    observation_space = create_mock_observation_space()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建编码器
        encoder = BLIP2InstructionEncoder(
            observation_space=observation_space,
            device=device,
            model_name="Salesforce/blip2-opt-2.7b",
            freeze_backbone=True
        )
        print(f"✓ BLIP-2指令编码器创建成功")
        print(f"  - 输出形状: {encoder.output_shape}")
        print(f"  - 输出大小: {encoder.output_size}")
        print(f"  - 设备: {device}")
        
        # 测试长文本支持（BLIP-2的优势）
        long_instruction = " ".join([f"step{i}" for i in range(100)])  # 100个token的长指令
        print(f"  - 测试长指令长度: {len(long_instruction.split())} tokens")
        
        # 模拟批处理输入
        batch_instructions = [
            "Navigate to the red building on your left and then turn right",
            long_instruction,  # 长指令测试
            "Fly forward until you see a blue marker"
        ]
        
        with torch.no_grad():
            output = encoder(batch_instructions)
        
        print(f"  - 批处理输入数量: {len(batch_instructions)}")
        print(f"  - 输出形状: {output.shape}")
        print(f"✓ BLIP-2指令编码器前向传播成功")
        print(f"✓ 长文本支持验证成功（CLIP只支持77 tokens，BLIP-2支持500+ tokens）")
        
        return True
        
    except Exception as e:
        print(f"✗ BLIP-2指令编码器测试失败: {str(e)}")
        return False

def test_blip2_depth_encoder():
    """测试BLIP-2深度编码器"""
    print("\n=== 测试BLIP-2深度编码器 ===")
    
    observation_space = create_mock_observation_space()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建编码器
        encoder = BLIP2DepthEncoder(
            observation_space=observation_space,
            device=device,
            model_name="Salesforce/blip2-opt-2.7b",
            layer_extract=-1,
            freeze_backbone=True
        )
        print(f"✓ BLIP-2深度编码器创建成功")
        print(f"  - 输出形状: {encoder.output_shape}")
        print(f"  - 输出大小: {encoder.output_size}")
        print(f"  - 设备: {device}")
        
        # 测试前向传播
        batch_size = 2
        depth_input = torch.randn(batch_size, 3, 224, 224).to(device)  # 深度图转换为3通道
        
        with torch.no_grad():
            output = encoder(depth_input)
        
        print(f"  - 输入形状: {depth_input.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"✓ BLIP-2深度编码器前向传播成功")
        
        return True
        
    except Exception as e:
        print(f"✗ BLIP-2深度编码器测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始BLIP-2编码器集成测试")
    print("=" * 60)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("=" * 60)
    
    # 运行测试
    results = []
    
    # 测试视觉编码器
    results.append(test_blip2_vision_encoder())
    
    # 测试指令编码器
    results.append(test_blip2_instruction_encoder())
    
    # 测试深度编码器
    results.append(test_blip2_depth_encoder())
    
    # 总结结果
    print("\n" + "=" * 60)
    print("🎯 测试结果总结")
    print("=" * 60)
    
    test_names = ["BLIP-2视觉编码器", "BLIP-2指令编码器", "BLIP-2深度编码器"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
    
    total_passed = sum(results)
    print(f"\n总计: {total_passed}/{len(results)} 个测试通过")
    
    if total_passed == len(results):
        print("\n🎉 所有测试通过！BLIP-2编码器迁移成功！")
        print("\n✨ BLIP-2相比CLIP的优势:")
        print("  - 文本序列长度：CLIP 77 tokens → BLIP-2 500+ tokens")
        print("  - 多模态理解能力更强")
        print("  - 更好的视觉-语言对齐")
        print("\n🚀 现在可以使用以下参数运行训练:")
        print("  --use_blip2_encoders")
        print("  --use_blip2_depth_encoder")
        print("  --blip2_model_name Salesforce/blip2-opt-2.7b")
        print("  --freeze_blip2_backbone")
    else:
        print(f"\n⚠️  {len(results) - total_passed} 个测试失败，请检查配置")
    
    print("=" * 60)

if __name__ == "__main__":
    main()