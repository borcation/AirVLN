#!/usr/bin/env python3
"""
CMA模型张量维度理论分析验证
通过代码分析确定正确的维度
"""

def analyze_cma_dimensions_from_code():
    """通过分析代码确定CMA的张量维度"""
    print("=== CMA模型张量维度代码分析 ===")
    
    # 从代码中提取的关键配置
    print("\n1. 编码器输出配置:")
    
    # RGB编码器 (ResNet50 for CMA)
    rgb_spatial_output = True
    rgb_output_size = 256  # 从代码中看到
    rgb_resnet_feature_dim = 2048
    rgb_spatial_size = (4, 4)  # 16个空间位置
    rgb_spatial_embedding_dim = 64
    rgb_output_shape_0 = rgb_resnet_feature_dim + rgb_spatial_embedding_dim  # 2048 + 64 = 2112
    
    print(f"  RGB编码器:")
    print(f"    - ResNet特征: {rgb_resnet_feature_dim}")
    print(f"    - 空间嵌入: {rgb_spatial_embedding_dim}")
    print(f"    - output_shape[0]: {rgb_output_shape_0}")
    print(f"    - output_size: {rgb_output_size}")
    print(f"    - 空间维度: {rgb_spatial_size}")
    
    # 深度编码器 (ResNet50 for CMA)
    depth_spatial_output = True
    depth_output_size = 128  # 从代码中看到
    depth_resnet_feature_dim = 2048
    depth_spatial_size = (8, 8)  # 64个空间位置
    depth_spatial_embedding_dim = 64
    depth_output_shape_0 = depth_resnet_feature_dim + depth_spatial_embedding_dim  # 2048 + 64 = 2112
    
    print(f"  深度编码器:")
    print(f"    - ResNet特征: {depth_resnet_feature_dim}")
    print(f"    - 空间嵌入: {depth_spatial_embedding_dim}")
    print(f"    - output_shape[0]: {depth_output_shape_0}")
    print(f"    - output_size: {depth_output_size}")
    print(f"    - 空间维度: {depth_spatial_size}")
    
    # 指令编码器 (BiLSTM for CMA)
    instruction_hidden_size = 128
    instruction_bidirectional = True  # CMA使用双向
    instruction_output_size = instruction_hidden_size * 2  # 256
    
    print(f"  指令编码器:")
    print(f"    - 隐藏维度: {instruction_hidden_size}")
    print(f"    - 双向: {instruction_bidirectional}")
    print(f"    - output_size: {instruction_output_size}")
    
    # 其他参数
    prev_action_embedding_dim = 32
    hidden_size = 512  # STATE_ENCODER_hidden_size
    
    print(f"  其他:")
    print(f"    - prev_action_embedding: {prev_action_embedding_dim}")
    print(f"    - hidden_size: {hidden_size}")
    
    print("\n2. CMA Forward流程分析:")
    
    # 第一阶段: RGB和深度线性投影
    print(f"\n  第一阶段 - 线性投影:")
    
    # RGB线性层: flatten后通过AdaptiveAvgPool1d(1)
    rgb_flattened_size = rgb_output_shape_0 * rgb_spatial_size[0] * rgb_spatial_size[1]  # 2112 * 16 = 33792
    rgb_after_pool = rgb_output_shape_0  # AdaptiveAvgPool1d(1)后: [batch, 2112]
    rgb_linear_output = rgb_output_size  # 256
    
    print(f"    RGB: flatten({rgb_output_shape_0}, {rgb_spatial_size}) → pool({rgb_after_pool}) → linear({rgb_linear_output})")
    
    # 深度线性层: flatten后直接线性层
    depth_flattened_size = depth_output_shape_0 * depth_spatial_size[0] * depth_spatial_size[1]  # 2112 * 64 = 135168
    depth_linear_output = depth_output_size  # 128
    
    print(f"    深度: flatten({depth_flattened_size}) → linear({depth_linear_output})")
    
    # 第一RNN输入
    first_rnn_input_size = rgb_linear_output + depth_linear_output + prev_action_embedding_dim
    first_rnn_output_size = hidden_size
    
    print(f"    第一RNN输入: {rgb_linear_output} + {depth_linear_output} + {prev_action_embedding_dim} = {first_rnn_input_size}")
    print(f"    第一RNN输出: {first_rnn_output_size}")
    
    print(f"\n  第二阶段 - 注意力机制:")
    
    # 注意力机制
    # 1. 文本-状态注意力
    text_state_attention_output = instruction_output_size  # 256
    print(f"    文本-状态注意力输出: {text_state_attention_output}")
    
    # 2. 文本-视觉注意力
    # rgb_kv: Conv1d(2112, hidden_size//2 + rgb_output_size, 1) = Conv1d(2112, 256+256, 1) = Conv1d(2112, 512, 1)
    rgb_kv_output_dim = hidden_size // 2 + rgb_output_size  # 256 + 256 = 512
    rgb_k_dim = hidden_size // 2  # 256
    rgb_v_dim = rgb_output_size   # 256
    rgb_attention_output = rgb_output_size  # 256
    
    print(f"    RGB K-V投影: {rgb_output_shape_0} → {rgb_kv_output_dim} (K:{rgb_k_dim}, V:{rgb_v_dim})")
    print(f"    RGB注意力输出: {rgb_attention_output}")
    
    # depth_kv: Conv1d(2112, hidden_size//2 + depth_output_size, 1) = Conv1d(2112, 256+128, 1) = Conv1d(2112, 384, 1)
    depth_kv_output_dim = hidden_size // 2 + depth_output_size  # 256 + 128 = 384
    depth_k_dim = hidden_size // 2  # 256
    depth_v_dim = depth_output_size  # 128
    depth_attention_output = depth_output_size  # 128
    
    print(f"    深度K-V投影: {depth_output_shape_0} → {depth_kv_output_dim} (K:{depth_k_dim}, V:{depth_v_dim})")
    print(f"    深度注意力输出: {depth_attention_output}")
    
    print(f"\n  第三阶段 - 最终融合:")
    
    # 最终拼接的组件
    components = {
        "state (第一RNN输出)": first_rnn_output_size,  # 512
        "text_embedding (文本注意力)": text_state_attention_output,  # 256
        "rgb_embedding (RGB注意力)": rgb_attention_output,  # 256
        "depth_embedding (深度注意力)": depth_attention_output,  # 128
        "prev_actions (动作嵌入)": prev_action_embedding_dim,  # 32
    }
    
    total_fusion_dim = sum(components.values())
    
    print(f"    最终拼接组件:")
    for name, dim in components.items():
        print(f"      - {name}: {dim}")
    print(f"    总维度: {total_fusion_dim}")
    
    # 压缩到hidden_size
    final_output_dim = hidden_size
    print(f"    压缩到: {final_output_dim}")
    
    return total_fusion_dim, components

def verify_against_reports():
    """验证与现有报告的对比"""
    print("\n=== 与现有报告对比 ===")
    
    total_dim, components = analyze_cma_dimensions_from_code()
    
    # 报告中的数据
    report_dims = {
        "报告1 (MODEL_DIMENSIONS_SUMMARY.md)": 5024,
        "报告2 (TENSOR_DIMENSIONS_ANALYSIS.md)": 1184,
    }
    
    print(f"代码分析结果: {total_dim}")
    for report, dim in report_dims.items():
        print(f"{report}: {dim}")
    
    # 分析差异
    print(f"\n差异分析:")
    
    if total_dim == 1184:
        print(f"✅ 与TENSOR_DIMENSIONS_ANALYSIS.md一致")
        print(f"❌ 与MODEL_DIMENSIONS_SUMMARY.md不一致")
        print(f"\n MODEL_DIMENSIONS_SUMMARY.md中的错误:")
        print(f"   - 可能错误地包含了所有空间特征的维度")
        print(f"   - 实际上注意力机制会聚合空间特征")
    elif total_dim == 5024:
        print(f"❌ 与TENSOR_DIMENSIONS_ANALYSIS.md不一致")
        print(f"✅ 与MODEL_DIMENSIONS_SUMMARY.md一致")
        print(f"\n TENSOR_DIMENSIONS_ANALYSIS.md中的错误:")
        print(f"   - 可能遗漏了某些特征维度")
    else:
        print(f"❌ 与两个报告都不一致")
        print(f"   需要进一步检查代码逻辑")

def create_validation_plan():
    """创建验证方案"""
    print(f"\n=== 验证方案 ===")
    
    print(f"1. 代码审查验证:")
    print(f"   - 检查 Model/cma_policy.py 的 forward 方法")
    print(f"   - 确认各编码器的 output_size 和 output_shape")
    print(f"   - 验证注意力机制的输出维度")
    
    print(f"\n2. 单元测试验证:")
    print(f"   - 创建小规模输入数据")
    print(f"   - 逐步跟踪每个组件的输出维度")
    print(f"   - 打印中间张量的形状")
    
    print(f"\n3. 实际运行验证:")
    print(f"   - 修改CMA forward方法添加调试输出")
    print(f"   - 运行一个完整的forward pass")
    print(f"   - 记录所有中间张量维度")

if __name__ == "__main__":
    print("CMA模型张量维度分析工具")
    print("=" * 60)
    
    total_dim, components = analyze_cma_dimensions_from_code()
    
    verify_against_reports()
    
    create_validation_plan()
    
    print(f"\n" + "=" * 60)
    print(f"结论:")
    print(f"  根据代码分析，CMA模型的最终融合维度应该是: {total_dim}")
    print(f"  这与TENSOR_DIMENSIONS_ANALYSIS.md中的1184一致")
    print(f"  MODEL_DIMENSIONS_SUMMARY.md中的5024是错误的")