#!/bin/bash

# 格式化时间为小时:分钟:秒
format_time() {
    local T=$1
    printf "%02d:%02d:%02d" $((T/3600)) $(( (T%3600)/60 )) $((T%60))
}

log_file="pipeline_time.log"
> $log_file  # 清空日志文件

run_and_log() {
    local script_name=$1
    local display_name=$2

    echo "开始运行 $script_name ..."
    local start_time=$(date +%s)
    bash $script_name
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local time_str=$(format_time $elapsed)
    local result="$display_name 阶段: $time_str"
    echo -e "$result"
    echo -e "$result" >> $log_file
}

# run_and_log AirVLN/scripts/collect_gpus.sh "collect_gpus"
# run_and_log AirVLN/scripts/collect_bs4_gpus.sh "collect_bs4_gpus"
# run_and_log AirVLN/scripts/collect_bs2_gpus.sh "collect_bs2_gpus"
# run_and_log AirVLN/scripts/collect_bs4.sh "collect_bs4"
# run_and_log AirVLN/scripts/collect_bs2.sh "collect_bs2"
# run_and_log AirVLN/scripts/collect.sh    "collect"

# run_and_log AirVLN/scripts/train_bs16_ep5.sh  "train_bs16_ep5"
# run_and_log AirVLN/scripts/train_bs16_ep10.sh "train_bs16_ep10"
# run_and_log AirVLN/scripts/train_bs16_ep20.sh "train_bs16_ep20"
# run_and_log AirVLN/scripts/train_bs8_ep5.sh  "train_bs8_ep5"
# run_and_log AirVLN/scripts/train_bs8_ep10.sh "train_bs8_ep10"
# run_and_log AirVLN/scripts/train_bs8_ep20.sh "train_bs8_ep20"
# run_and_log AirVLN/scripts/train_bs4_ep5.sh  "train_bs4_ep5"
# run_and_log AirVLN/scripts/train_bs4_ep10.sh  "train_bs4_ep10"
# run_and_log AirVLN/scripts/train_bs4_ep50.sh  "train_bs4_ep50"

# run_and_log AirVLN/scripts/eval.sh     "eval"

#测试collect_ws.sh脚本，batchsize可调，分别测试1，2，4，8，16
for bs in 1 2 4 8 16; do
    export BATCH_SIZE=$bs
    run_and_log AirVLN/scripts/collect_ws.sh "collect_ws_bs$bs"
done