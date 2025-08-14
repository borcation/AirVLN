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

run_and_log AirVLN/scripts/collect.sh        "collect"
run_and_log AirVLN/scripts/train_bs8_ep5.sh    "train_bs8_ep5"
run_and_log AirVLN/scripts/eval.sh    "eval"