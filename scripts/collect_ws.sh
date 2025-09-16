#!/bin/bash
# 7960工作站使用脚本

eval "$(conda shell.bash hook)"
ps -ef | grep AirVLN-Linux | grep -v grep | awk '{print $2}' | xargs kill -9  

lsof -ti:30000 | xargs -r kill -9

conda activate AirVLN

rm -rf ./DATA/img_features/collect/AirVLN-seq2seq  # 删除指定目录

cd ./AirVLN
echo $PWD

#设置使用的显卡编号，0,1,2,3表示使用4张显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

#注意，对于airsim程序，在有核显的机器上，gpu0通常是核显，gpu1才是独立显卡，请务必确认
#确认方法为手动运行以下指令，观察程序跑在哪儿
#/home/work/AirVLN_ws/ENVs/env_11/env_11/LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping AirVLN -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter=0 --settings /home/work/AirVLN_ws/AirVLN/airsim_plugin/settings/1/settings.json

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3 &

sleep 5

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize "${BATCH_SIZE:-1}"  # batchSize为外部参数BATCH_SIZE，默认值为1，可通过环境变量传入

#这里的batchsize是指每个显卡上运行的模拟器的个数
#收集的时候，一张显卡最好不要超过2个batch，batchsize会均分到每个显卡上

#经过测试，收集的最快的方式是batchsize开1，一条条收集。（约30+小时）
#如果batchsize开大，虽然每个显卡上同时运行多个模拟器，但是会导致每个模拟器的运行速度变慢，反而收集更慢。




