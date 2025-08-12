#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate AirVLN

cd ./AirVLN
echo $PWD

# 建议减少 CUDA 内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u ./src/vlnce_src/train.py \
--run_type train \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 4 \
--dagger_it 1 \
--epochs 50 \
--lr 0.00025 \
--trainer_gpu_device 0 \
--amp


# nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 &

# python -u ./src/vlnce_src/dagger_train.py \
# --run_type train \
# --policy_type seq2seq \
# --collect_type dagger \
# --name AirVLN-seq2seq-dagger \
# --batchSize 4 \
# --dagger_it 10 \
# --epochs 5 \
# --lr 0.00025 \
# --trainer_gpu_device 0 \
# --dagger_update_size 5000


