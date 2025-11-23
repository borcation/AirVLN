#!/bin/bash
# 评估脚本 - 使用BLIP-2编码器
# 用法: bash scripts/eval_blip2.sh

eval "$(conda shell.bash hook)"
conda activate AirVLN

# 评估配置
export EVAL_CKPT_PATH_DIR="/home/work/AirVLN_ws/DATA/output/AirVLN-cma-blip-1000/train/checkpoint"
export EVAL_DATASET="val_unseen"  # 可选: val_seen, val_unseen, test

python -u src/vlnce_src/train.py \
    --run_type eval \
    --policy_type cma \
    --name AirVLN-cma-blip-1000 \
    --use_blip2_encoders \
    --use_blip2_depth_encoder \
    --blip2_model_name Salesforce/blip2-opt-2.7b \
    --freeze_blip2_backbone \
    --batchSize 1 \
    --EVAL_CKPT_PATH_DIR ${EVAL_CKPT_PATH_DIR} \
    --EVAL_DATASET ${EVAL_DATASET}