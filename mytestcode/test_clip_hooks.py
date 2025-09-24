#!/usr/bin/env python3
"""
测试CLIP编码器的hook兼容性和GPU使用情况
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())

from Model.encoders.clip_encoder import CLIPVisionEncoder, CLIPDepthEncoder
from Model.encoders.resnet_encoders import TorchVisionResNet50, VlnResnetDepthEncoder
from gym import spaces

def test_encoder_hooks():
    """测试不同编码器的hook兼容性"""
    print("=== 编码器Hook兼容性测试 ===")
    
    # 创建观测空间
    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=torch.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=torch.float32),
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试ResNet编码器
    print("\n1. 测试ResNet RGB编码器:")
    try:
        resnet_rgb = TorchVisionResNet50(observation_space, device)
        print(f"   ResNet RGB编码器属性:")
        print(f"   - 有layer_extract: {hasattr(resnet_rgb, 'layer_extract')}")
        if hasattr(resnet_rgb, 'layer_extract'):
            print(f"   - layer_extract类型: {type(resnet_rgb.layer_extract)}")
            print(f"   - layer_extract: {resnet_rgb.layer_extract}")
    except Exception as e:
        print(f"   ResNet RGB编码器测试失败: {e}")
    
    print("\n2. 测试ResNet深度编码器:")
    try:
        resnet_depth = VlnResnetDepthEncoder(observation_space)
        print(f"   ResNet深度编码器属性:")
        print(f"   - 有visual_encoder: {hasattr(resnet_depth, 'visual_encoder')}")
        if hasattr(resnet_depth, 'visual_encoder'):
            print(f"   - visual_encoder类型: {type(resnet_depth.visual_encoder)}")
            print(f"   - visual_encoder有layer_extract: {hasattr(resnet_depth.visual_encoder, 'layer_extract') if hasattr(resnet_depth, 'visual_encoder') else False}")
    except Exception as e:
        print(f"   ResNet深度编码器测试失败: {e}")
    
    # 测试CLIP编码器
    print("\n3. 测试CLIP RGB编码器:")
    try:
        clip_rgb = CLIPVisionEncoder(observation_space, device)
        print(f"   CLIP RGB编码器属性:")
        print(f"   - 有layer_extract: {hasattr(clip_rgb, 'layer_extract')}")
        print(f"   - 有vision_model: {hasattr(clip_rgb, 'vision_model')}")
        
        # 查看CLIP内部结构
        if hasattr(clip_rgb, 'vision_model'):
            print(f"   - vision_model类型: {type(clip_rgb.vision_model)}")
            vision_model = clip_rgb.vision_model
            print(f"   - vision_model的子模块:")
            for name, module in vision_model.named_children():
                print(f"     - {name}: {type(module)}")
        
        # 尝试找到合适的hook点
        print(f"   - 可能的hook点:")
        for name, module in clip_rgb.named_modules():
            if 'pooler' in name or 'pool' in name or 'projection' in name:
                print(f"     - {name}: {type(module)}")
                
    except Exception as e:
        print(f"   CLIP RGB编码器测试失败: {e}")
    
    print("\n4. 测试CLIP深度编码器:")
    try:
        clip_depth = CLIPDepthEncoder(observation_space, device)
        print(f"   CLIP深度编码器属性:")
        print(f"   - 有layer_extract: {hasattr(clip_depth, 'layer_extract')}")
        print(f"   - 有vision_model: {hasattr(clip_depth, 'vision_model')}")
    except Exception as e:
        print(f"   CLIP深度编码器测试失败: {e}")

def test_hook_registration():
    """测试hook注册机制"""
    print("\n=== Hook注册测试 ===")
    
    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=torch.uint8),
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def hook_builder(tgt_tensor):
        def hook(m, i, o):
            print(f"Hook触发: 模块={type(m)}, 输出形状={o.shape if hasattr(o, 'shape') else type(o)}")
            tgt_tensor.set_(o.cpu() if hasattr(o, 'cpu') else o)
        return hook
    
    # 测试ResNet hook
    print("\n1. 测试ResNet hook:")
    try:
        resnet_encoder = TorchVisionResNet50(observation_space, device)
        rgb_features = torch.zeros((1,), device="cpu")
        
        if hasattr(resnet_encoder, 'layer_extract'):
            hook = resnet_encoder.layer_extract.register_forward_hook(hook_builder(rgb_features))
            print("   ResNet hook注册成功")
            
            # 创建测试输入
            test_input = {
                "rgb": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8).to(device)
            }
            
            with torch.no_grad():
                output = resnet_encoder(test_input)
                print(f"   ResNet输出形状: {output.shape}")
                print(f"   提取的特征形状: {rgb_features.shape}")
            
            hook.remove()
        else:
            print("   ResNet没有layer_extract属性")
            
    except Exception as e:
        print(f"   ResNet hook测试失败: {e}")
    
    # 测试CLIP hook (需要手动添加)
    print("\n2. 测试CLIP hook (手动添加layer_extract):")
    try:
        clip_encoder = CLIPVisionEncoder(observation_space, device)
        
        # 手动添加layer_extract指向合适的层
        if hasattr(clip_encoder, 'vision_model'):
            # 尝试指向vision_model的pooler或最后一层
            if hasattr(clip_encoder.vision_model, 'pooler'):
                clip_encoder.layer_extract = clip_encoder.vision_model.pooler
                print("   为CLIP添加了layer_extract (指向pooler)")
            else:
                # 如果没有pooler，指向vision_model本身
                clip_encoder.layer_extract = clip_encoder.vision_model
                print("   为CLIP添加了layer_extract (指向vision_model)")
            
            rgb_features = torch.zeros((1,), device="cpu")
            hook = clip_encoder.layer_extract.register_forward_hook(hook_builder(rgb_features))
            print("   CLIP hook注册成功")
            
            # 创建测试输入
            test_input = {
                "rgb": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8).to(device)
            }
            
            with torch.no_grad():
                output = clip_encoder(test_input)
                print(f"   CLIP输出形状: {output.shape}")
                print(f"   提取的特征形状: {rgb_features.shape}")
            
            hook.remove()
        else:
            print("   CLIP没有vision_model属性")
            
    except Exception as e:
        print(f"   CLIP hook测试失败: {e}")

def test_gpu_usage():
    """测试GPU使用情况"""
    print("\n=== GPU使用情况测试 ===")
    
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        
        # 测试内存使用
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        print(f"初始GPU内存使用: {initial_memory / 1024**2:.2f} MB")
        
        observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=torch.uint8),
        })
        device = torch.device("cuda")
        
        # 创建编码器并测试内存使用
        print("\n测试编码器GPU内存使用:")
        
        # ResNet
        try:
            resnet_encoder = TorchVisionResNet50(observation_space, device)
            resnet_memory = torch.cuda.memory_allocated()
            print(f"ResNet编码器内存: {(resnet_memory - initial_memory) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"ResNet内存测试失败: {e}")
        
        # CLIP
        try:
            clip_encoder = CLIPVisionEncoder(observation_space, device)
            clip_memory = torch.cuda.memory_allocated()
            print(f"CLIP编码器内存: {(clip_memory - resnet_memory) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"CLIP内存测试失败: {e}")
        
        torch.cuda.empty_cache()
    else:
        print("CUDA不可用，只能使用CPU")

if __name__ == "__main__":
    print("AirVLN 编码器Hook兼容性和GPU使用分析")
    print("=" * 60)
    
    test_encoder_hooks()
    test_hook_registration()
    test_gpu_usage()
    
    print("\n" + "=" * 60)
    print("测试完成！")