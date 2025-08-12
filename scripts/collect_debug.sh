#!/bin/bash

eval "$(conda shell.bash hook)"
ps -ef | grep AirVLN-Linux | grep -v grep | awk '{print $2}' | xargs kill -9  

lsof -ti:30000 | xargs -r kill -9


conda activate AirVLN

rm -rf ./DATA/img_features/collect/AirVLN-seq2seq  # 删除指定目录


cd ./AirVLN
echo $PWD

export CUDA_VISIBLE_DEVICES=0

python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 1 &


