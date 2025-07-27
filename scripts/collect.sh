
conda activate AirVLN

cd ./AirVLN
echo $PWD

export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/utils

export CUDA_VISIBLE_DEVICES=0

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 2

