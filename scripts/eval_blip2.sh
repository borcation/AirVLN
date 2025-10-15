#!/bin/bash
# 评估脚本 - 使用BLIP-2编码器
# 用法: bash scripts/eval_blip2.sh

python -u train.py \
    --run_type eval \
    --policy_type cma \
    --name AirVLN-cma-blip2 \
    --use_blip2_encoders \
    --use_blip2_depth_encoder \
    --blip2_model_name Salesforce/blip2-opt-2.7b \
    --freeze_blip2_backbone \
    --batchSize 1