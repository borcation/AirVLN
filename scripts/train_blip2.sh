#!/bin/bash
# 训练脚本 - 使用BLIP-2编码器
# 用法: bash scripts/train_blip2.sh

eval "$(conda shell.bash hook)"
conda activate AirVLN

# 优化显存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# BLIP-2相关参数
export USE_BLIP2_ENCODERS="--use_blip2_encoders"
export USE_BLIP2_DEPTH_ENCODER="--use_blip2_depth_encoder"  # 同时对深度图像使用BLIP-2
export BLIP2_MODEL_NAME="--blip2_model_name Salesforce/blip2-opt-2.7b"
export FREEZE_BLIP2_BACKBONE="--freeze_blip2_backbone"  # 可选：冻结BLIP-2 backbone

# 训练配置
export EPOCHS="--epochs 20"
export BATCH_SIZE="--batchSize 1"

# 使用 torchrun 启动多卡训练
torchrun --nproc_per_node=4 --master_port=29500 src/vlnce_src/train.py \
    --policy_type cma \
    --run_type train \
    --name AirVLN-cma-blip-1000 \
    ${USE_BLIP2_ENCODERS} \
    ${USE_BLIP2_DEPTH_ENCODER} \
    ${BLIP2_MODEL_NAME} \
    ${FREEZE_BLIP2_BACKBONE} \
    ${EPOCHS} \
    ${BATCH_SIZE} \
    --lr 1e-5 \
    --collect_type TF \
    --DistributedDataParallel