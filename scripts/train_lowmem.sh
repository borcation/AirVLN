#!/bin/bash

set -euo pipefail

# 激活 conda 环境（与仓库现有脚本一致）
eval "$(conda shell.bash hook)"
conda activate AirVLN

# 建议减少 CUDA 内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ./AirVLN

echo "CWD: $PWD"

python -u ./src/vlnce_src/train.py \
  --run_type train \
  --policy_type seq2seq \
  --collect_type TF \
  --name AirVLN-seq2seq \
  --batchSize 1 \
  --dagger_it 1 \
  --epochs 5 \
  --lr 0.00025 \
  --trainer_gpu_device 0 \
  --maxAction 500 \
  --amp \
  # --ablate_rgb

# 说明：
# 1) --amp 启用混合精度，显存占用明显降低，训练时间还能减少10%-30%。
# 2) --maxAction 120 将RNN缓存序列长度上限裁剪到 120（默认最大为 500），降低部分时间维度开销。
# 3) --ablate_rgb 关闭 RGB 编码器前向，以进一步降显存/算力。若希望保留 RGB，可去掉该标志再试。
