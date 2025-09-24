#!/bin/bash
# 对比不同编码器配置的训练脚本

cd /home/work/AirVLN_ws/AirVLN

echo "=========================================="
echo "AirVLN CLIP编码器训练配置选项"
echo "=========================================="

# 配置1: 原始ResNet + BiLSTM (基线)
echo "配置1: 原始编码器 (ResNet50 + BiLSTM)"
echo "python -u -m src.vlnce_src.run \\"
echo "    --run_type train \\"
echo "    --policy_type cma \\"
echo "    --batchSize 8 \\"
echo "    --epochs 5 \\"
echo "    --name AirVLN-cma-baseline"
echo ""

# 配置2: CLIP RGB + Text, 保留ResNet Depth
echo "配置2: CLIP图像和文本编码器 (保留ResNet深度编码器)"
echo "python -u -m src.vlnce_src.run \\"
echo "    --run_type train \\"
echo "    --policy_type cma \\"
echo "    --use_clip_encoders \\"
echo "    --freeze_clip_backbone \\"
echo "    --batchSize 4 \\"
echo "    --epochs 5 \\"
echo "    --name AirVLN-cma-clip-rgb-text"
echo ""

# 配置3: 完全CLIP编码器 (RGB + Depth + Text)
echo "配置3: 完全CLIP编码器 (RGB + 深度 + 文本)"
echo "python -u -m src.vlnce_src.run \\"
echo "    --run_type train \\"
echo "    --policy_type cma \\"
echo "    --use_clip_encoders \\"
echo "    --use_clip_depth_encoder \\"
echo "    --freeze_clip_backbone \\"
echo "    --batchSize 4 \\"
echo "    --epochs 5 \\"
echo "    --name AirVLN-cma-clip-full"
echo ""

# 配置4: CLIP编码器 + Fine-tuning
echo "配置4: CLIP编码器微调 (不冻结backbone)"
echo "python -u -m src.vlnce_src.run \\"
echo "    --run_type train \\"
echo "    --policy_type cma \\"
echo "    --use_clip_encoders \\"
echo "    --use_clip_depth_encoder \\"
echo "    --batchSize 2 \\"
echo "    --lr 0.0001 \\"
echo "    --epochs 10 \\"
echo "    --name AirVLN-cma-clip-finetune"
echo ""

echo "使用说明:"
echo "1. 从配置1开始建立基线性能"
echo "2. 运行配置2测试RGB+文本CLIP的效果"
echo "3. 运行配置3测试完全CLIP编码器"
echo "4. 如果效果好，尝试配置4进行精细调优"
echo ""
echo "内存使用建议:"
echo "- 配置1: 可用较大batch size (8-16)"
echo "- 配置2-3: 中等batch size (4-8)" 
echo "- 配置4: 较小batch size (2-4)"
echo ""

read -p "选择要运行的配置 (1-4): " config_choice

case $config_choice in
    1)
        echo "运行配置1: 原始编码器..."
        python -u -m src.vlnce_src.run \
            --run_type train \
            --policy_type cma \
            --batchSize 8 \
            --epochs 5 \
            --name AirVLN-cma-baseline \
            --amp
        ;;
    2)
        echo "运行配置2: CLIP RGB+文本编码器..."
        python -u -m src.vlnce_src.run \
            --run_type train \
            --policy_type cma \
            --use_clip_encoders \
            --freeze_clip_backbone \
            --batchSize 4 \
            --epochs 5 \
            --name AirVLN-cma-clip-rgb-text \
            --amp
        ;;
    3)
        echo "运行配置3: 完全CLIP编码器..."
        python -u -m src.vlnce_src.run \
            --run_type train \
            --policy_type cma \
            --use_clip_encoders \
            --use_clip_depth_encoder \
            --freeze_clip_backbone \
            --batchSize 4 \
            --epochs 5 \
            --name AirVLN-cma-clip-full \
            --amp
        ;;
    4)
        echo "运行配置4: CLIP编码器微调..."
        python -u -m src.vlnce_src.run \
            --run_type train \
            --policy_type cma \
            --use_clip_encoders \
            --use_clip_depth_encoder \
            --batchSize 2 \
            --lr 0.0001 \
            --epochs 10 \
            --name AirVLN-cma-clip-finetune \
            --amp
        ;;
    *)
        echo "无效选择，退出..."
        exit 1
        ;;
esac