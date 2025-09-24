#!/bin/bash
# CLIP编码器评估脚本示例

cd /home/work/AirVLN_ws/AirVLN

# 评估使用CLIP编码器的模型（包括深度编码器）
python -u -m src.vlnce_src.run \
    --run_type eval \
    --policy_type cma \
    --batchSize 1 \
    --name AirVLN-cma-clip-full \
    --use_clip_encoders \
    --use_clip_depth_encoder \
    --clip_model_name openai/clip-vit-base-patch32 \
    --freeze_clip_backbone \
    --EVAL_CKPT_PATH_DIR DATA/output/AirVLN-cma-clip-full/train/checkpoints \
    --EVAL_DATASET val_unseen \
    --EVAL_NUM 100