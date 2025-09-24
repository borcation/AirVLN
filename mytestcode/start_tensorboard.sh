#!/bin/bash

echo "=== AirVLN TensorBoard 启动脚本 ==="
echo "训练数据: AirVLN-seq2seq-s (2025年9月22日)"
echo ""

# 设置路径
TENSORBOARD_DIR="/home/work/AirVLN_ws/DATA/output/AirVLN-seq2seq-s/train/TensorBoard"
PORT=6006

echo "TensorBoard 日志目录: $TENSORBOARD_DIR"
echo "端口: $PORT"
echo ""

# 检查目录是否存在
if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo "❌ 错误: TensorBoard 目录不存在"
    echo "   路径: $TENSORBOARD_DIR"
    exit 1
fi

echo "✅ 找到 TensorBoard 日志目录"

# 激活conda环境并启动TensorBoard
echo "正在启动 TensorBoard..."
echo "请在浏览器中打开: http://localhost:$PORT"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo "=========================================="

# 尝试激活conda环境
if command -v conda &> /dev/null; then
    echo "激活 conda 环境: AirVLN"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate AirVLN 2>/dev/null
fi

# 启动TensorBoard
tensorboard --logdir="$TENSORBOARD_DIR" --port=$PORT --host=0.0.0.0