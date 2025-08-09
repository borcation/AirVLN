#!/bin/bash

eval "$(conda shell.bash hook)"
ps -ef | grep AirVLN-Linux | grep -v grep | awk '{print $2}' | xargs kill -9  

lsof -ti:30000 | xargs -r kill -9


conda activate AirVLN

rm -rf ./DATA/img_features/collect/AirVLN-seq2seq  # 删除指定目录


cd ./AirVLN
echo $PWD

export CUDA_VISIBLE_DEVICES=0

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 1 &

sleep 5

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 4


