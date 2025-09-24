#!/bin/bash
# CLIP编码器训练脚本示例
# 基于原始train.sh但添加了CLIP编码器参数

cd /home/work/AirVLN_ws/AirVLN

# 设置CLIP编码器参数
export USE_CLIP_ENCODERS="--use_clip_encoders"
export USE_CLIP_DEPTH_ENCODER="--use_clip_depth_encoder"  # 同时对深度图像使用CLIP
export CLIP_MODEL_NAME="--clip_model_name openai/clip-vit-base-patch32"
export FREEZE_CLIP_BACKBONE="--freeze_clip_backbone"  # 可选：冻结CLIP backbone

python -u -m src.vlnce_src.run \
    --run_type train \
    --policy_type cma \
    --batchSize 4 \
    --lr 0.00025 \
    --epochs 5 \
    --name AirVLN-cma-clip-full \
    ${USE_CLIP_ENCODERS} \
    ${USE_CLIP_DEPTH_ENCODER} \
    ${CLIP_MODEL_NAME} \
    ${FREEZE_CLIP_BACKBONE} \
    --amp