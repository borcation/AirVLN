#!/bin/bash
# 训练脚本 - 使用BLIP-2编码器
# 用法: bash scripts/train_blip2.sh

# BLIP-2相关参数
export USE_BLIP2_ENCODERS="--use_blip2_encoders"
export USE_BLIP2_DEPTH_ENCODER="--use_blip2_depth_encoder"  # 同时对深度图像使用BLIP-2
export BLIP2_MODEL_NAME="--blip2_model_name Salesforce/blip2-opt-2.7b"
export FREEZE_BLIP2_BACKBONE="--freeze_blip2_backbone"  # 可选：冻结BLIP-2 backbone

# 训练配置
export EPOCHS="--epochs 20"
export BATCH_SIZE="--batchSize 8"

python -u train.py \
    --policy_type cma \
    --run_type train \
    --name AirVLN-cma-blip2 \
    ${USE_BLIP2_ENCODERS} \
    ${USE_BLIP2_DEPTH_ENCODER} \
    ${BLIP2_MODEL_NAME} \
    ${FREEZE_BLIP2_BACKBONE} \
    ${EPOCHS} \
    ${BATCH_SIZE} \
    --lr 1e-5 \
    --action_embed_size 2048 \
    --feedback tf