

eval "$(conda shell.bash hook)"
ps -ef | grep AirVLN-Linux | grep -v grep | awk '{print $2}' | xargs kill -9  

lsof -ti:30000 | xargs -r kill -9


conda activate AirVLN

cd ./AirVLN
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &




