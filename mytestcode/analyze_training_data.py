#!/usr/bin/env python3
"""
AirVLN 训练数据分析工具
用于查看checkpoint和TensorBoard数据
"""

import torch
import os
import sys
from collections import OrderedDict
import numpy as np

def analyze_checkpoint(ckpt_path):
    """分析checkpoint文件"""
    print(f"\n=== 分析 Checkpoint: {os.path.basename(ckpt_path)} ===")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"Checkpoint 包含的键: {list(checkpoint.keys())}")
        
        # 分析模型状态
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n模型参数数量: {len(state_dict)}")
            
            # 按模块分组分析
            modules = {}
            total_params = 0
            
            for name, param in state_dict.items():
                if param.numel() > 0:
                    total_params += param.numel()
                
                # 提取模块名
                module_name = name.split('.')[0] if '.' in name else name
                if module_name not in modules:
                    modules[module_name] = []
                modules[module_name].append((name, param.shape, param.numel()))
            
            print(f"总参数数量: {total_params:,}")
            
            # 显示各模块的参数
            print("\n=== 模块参数详情 ===")
            for module, params in modules.items():
                module_total = sum(p[2] for p in params)
                print(f"\n{module}: {module_total:,} 参数")
                
                # 显示前5个参数的维度
                for i, (name, shape, numel) in enumerate(params[:5]):
                    print(f"  {name}: {shape} ({numel:,} 参数)")
                
                if len(params) > 5:
                    print(f"  ... 还有 {len(params)-5} 个参数")
        
        # 分析训练信息
        if 'step' in checkpoint:
            print(f"\n训练步数: {checkpoint['step']}")
        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch']}")
        if 'optim_state' in checkpoint:
            print(f"优化器状态已保存")
            
        # 查看损失信息
        loss_keys = [k for k in checkpoint.keys() if 'loss' in k.lower()]
        if loss_keys:
            print(f"\n损失相关信息: {loss_keys}")
            for key in loss_keys:
                if isinstance(checkpoint[key], (int, float)):
                    print(f"  {key}: {checkpoint[key]}")
        
    except Exception as e:
        print(f"加载checkpoint失败: {e}")

def get_model_architecture_info(ckpt_path):
    """推断模型架构信息"""
    print(f"\n=== 推断模型架构 ===")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' not in checkpoint:
            print("未找到模型状态字典")
            return
            
        state_dict = checkpoint['state_dict']
        
        # 查找编码器相关参数
        encoder_info = {
            'instruction_encoder': [],
            'rgb_encoder': [],
            'depth_encoder': [],
            'other': []
        }
        
        for name, param in state_dict.items():
            if 'instruction' in name.lower():
                encoder_info['instruction_encoder'].append((name, param.shape))
            elif 'rgb' in name.lower() or 'visual' in name.lower():
                encoder_info['rgb_encoder'].append((name, param.shape))
            elif 'depth' in name.lower():
                encoder_info['depth_encoder'].append((name, param.shape))
            else:
                encoder_info['other'].append((name, param.shape))
        
        # 显示编码器信息
        for encoder_type, params in encoder_info.items():
            if params:
                print(f"\n{encoder_type.upper()}:")
                for name, shape in params[:3]:  # 只显示前3个
                    print(f"  {name}: {shape}")
                if len(params) > 3:
                    print(f"  ... 还有 {len(params)-3} 个参数")
        
        # 尝试推断编码器类型
        print(f"\n=== 编码器类型推断 ===")
        
        # 查找特征维度线索
        feature_dims = []
        for name, param in state_dict.items():
            if 'linear' in name.lower() or 'fc' in name.lower():
                if len(param.shape) == 2:  # 线性层
                    feature_dims.append(f"{name}: {param.shape}")
        
        if feature_dims:
            print("线性层维度（可能反映特征维度）:")
            for dim in feature_dims[:5]:
                print(f"  {dim}")
                
    except Exception as e:
        print(f"分析模型架构失败: {e}")

def analyze_tensorboard_data():
    """分析TensorBoard数据"""
    print(f"\n=== TensorBoard 数据分析 ===")
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        tb_path = "/home/work/AirVLN_ws/DATA/output/AirVLN-seq2seq-s/train/TensorBoard/20250922-181721-497138"
        
        # 创建事件累积器
        event_acc = EventAccumulator(tb_path)
        event_acc.Reload()
        
        # 获取所有标量数据的键
        scalar_keys = event_acc.Tags()['scalars']
        print(f"可用的标量数据: {scalar_keys}")
        
        # 显示一些关键指标
        key_metrics = ['loss', 'accuracy', 'learning_rate', 'reward']
        
        for metric in key_metrics:
            matching_keys = [k for k in scalar_keys if metric.lower() in k.lower()]
            if matching_keys:
                print(f"\n{metric.upper()} 相关指标:")
                for key in matching_keys:
                    scalar_events = event_acc.Scalars(key)
                    if scalar_events:
                        latest = scalar_events[-1]
                        print(f"  {key}: 最新值 = {latest.value:.4f} (步数 {latest.step})")
        
        # 检查是否有直方图数据（可能包含张量维度信息）
        if 'histograms' in event_acc.Tags():
            hist_keys = event_acc.Tags()['histograms']
            print(f"\n直方图数据: {hist_keys}")
        
        # 检查是否有图像数据
        if 'images' in event_acc.Tags():
            image_keys = event_acc.Tags()['images']
            print(f"\n图像数据: {image_keys}")
            
    except ImportError:
        print("需要安装 tensorboard 来分析 TensorBoard 数据")
        print("请运行: pip install tensorboard")
    except Exception as e:
        print(f"分析 TensorBoard 数据失败: {e}")

def main():
    print("AirVLN 训练数据分析工具")
    print("=" * 50)
    
    # 分析最新的checkpoint
    ckpt_dir = "/home/work/AirVLN_ws/DATA/output/AirVLN-seq2seq-s/train/checkpoint/20250922-181721-497138"
    
    # 找到最新的checkpoint
    latest_ckpt = os.path.join(ckpt_dir, "ckpt.LAST.pth")
    if os.path.exists(latest_ckpt):
        analyze_checkpoint(latest_ckpt)
        get_model_architecture_info(latest_ckpt)
    else:
        print(f"未找到最新checkpoint: {latest_ckpt}")
        # 尝试分析其他checkpoint
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if ckpt_files:
            # 分析数值最大的checkpoint
            numbered_ckpts = []
            for f in ckpt_files:
                try:
                    if f.startswith('ckpt.') and f != 'ckpt.LAST.pth':
                        num = int(f.split('.')[1])
                        numbered_ckpts.append((num, f))
                except:
                    continue
            
            if numbered_ckpts:
                latest_num_ckpt = max(numbered_ckpts)[1]
                latest_path = os.path.join(ckpt_dir, latest_num_ckpt)
                analyze_checkpoint(latest_path)
                get_model_architecture_info(latest_path)
    
    # 分析TensorBoard数据
    analyze_tensorboard_data()
    
    print(f"\n=== 关于张量维度变化 ===")
    print("Checkpoint文件主要包含:")
    print("1. 模型参数权重 - 可以看到各层的维度")
    print("2. 优化器状态 - 包含梯度信息")
    print("3. 训练步数和损失值")
    print("\nTensorBoard日志主要包含:")
    print("1. 标量指标（损失、准确率等）的变化曲线")
    print("2. 可能的直方图数据（参数分布）")
    print("3. 学习率变化等")
    print("\n要查看运行时张量维度变化，需要在代码中添加调试输出或使用调试器。")

if __name__ == "__main__":
    main()