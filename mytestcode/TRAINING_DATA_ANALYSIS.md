# AirVLN 训练数据分析报告

## 数据概览

您的训练数据位于：
- **Checkpoint目录**: `DATA/output/AirVLN-seq2seq-s/train/checkpoint/20250922-181721-497138/`
- **TensorBoard目录**: `DATA/output/AirVLN-seq2seq-s/train/TensorBoard/20250922-181721-497138/`

## 1. Checkpoint 数据分析

### 基本信息
- **训练时间**: 2025年9月22日 18:17:21 开始
- **最新checkpoint**: `ckpt.LAST.pth` (对应第499个epoch)
- **模型总参数**: 35,641,042 个参数
- **训练完成度**: 499个epoch (完整训练)

### 模型结构分析
基于checkpoint参数名称，可以确认这是标准的Seq2Seq模型：

```
指令编码器 (BiLSTM):
- net.instruction_encoder.encoder_rnn.weight_ih_l0: [512, 50]
- net.instruction_encoder.encoder_rnn.weight_hh_l0: [512, 128]  
- net.instruction_encoder.embedding_layer.weight: [10038, 50]

深度编码器 (ResNet):
- net.depth_encoder.visual_encoder.backbone.conv1.0.weight: [32, 1, 7, 7]
- 总共有完整的ResNet结构参数

RGB编码器 (ResNet):
- 类似的ResNet结构参数
```

### 推断的维度信息
从参数形状可以推断：
- **词汇表大小**: 10,038 (embedding层)
- **词嵌入维度**: 50
- **LSTM隐藏层**: 128维
- **输入图像**: 3通道RGB + 1通道深度

## 2. TensorBoard 数据分析

### 日志文件
- **事件文件**: `events.out.tfevents.1758536242.Dell-7960-Tower`
- **可查看内容**:
  - 训练损失曲线
  - 验证指标变化
  - 学习率调度
  - 可能的参数分布直方图

### 如何查看TensorBoard
```bash
# 启动TensorBoard服务器
tensorboard --logdir=DATA/output/AirVLN-seq2seq-s/train/TensorBoard/20250922-181721-497138

# 在浏览器打开
http://localhost:6006
```

## 3. 关于张量维度变化的说明

### ✅ 可以从数据中获得的信息：
1. **模型架构**: 通过checkpoint参数名和形状
2. **训练进度**: 通过TensorBoard的损失曲线
3. **参数数量**: 各层的参数规模
4. **静态维度**: 各层的输入输出维度（从权重形状推断）

### ❌ 数据中没有的信息：
1. **运行时张量维度变化**: 前向传播过程中每一步的张量形状
2. **动态维度信息**: batch_size、序列长度等运行时确定的维度
3. **中间激活值的形状**: 各层输出的实际张量维度
4. **注意力权重分布**: 注意力机制的实际权重值

## 4. 如何查看运行时张量维度

要查看模型运行过程中的张量维度变化，需要在代码中添加调试输出：

### 方法1: 在模型代码中添加打印语句
```python
# 在 Model/seq2seq_policy.py 中添加
def forward(self, observations, rnn_hidden_states, prev_actions, masks):
    print(f"Input observations shape: {observations['rgb'].shape}")
    
    instruction_embedding = self.instruction_encoder(observations["instruction"])
    print(f"Instruction embedding shape: {instruction_embedding.shape}")
    
    rgb_features = self.rgb_encoder(observations["rgb"])
    print(f"RGB features shape: {rgb_features.shape}")
    
    # ... 继续为每个关键步骤添加打印
```

### 方法2: 使用PyTorch hooks
```python
def print_tensor_hook(name):
    def hook(module, input, output):
        print(f"{name} - Input: {input[0].shape if input else None}")
        print(f"{name} - Output: {output.shape if hasattr(output, 'shape') else type(output)}")
    return hook

# 注册hooks
model.rgb_encoder.register_forward_hook(print_tensor_hook("RGB_Encoder"))
model.instruction_encoder.register_forward_hook(print_tensor_hook("Instruction_Encoder"))
```

### 方法3: 使用调试模式运行
```bash
# 在评估脚本中添加 --debug 参数来输出维度信息
python -u run.py --exp-config configs/experiments/seq2seq_debug.yaml --debug-mode
```

## 5. 建议的下一步

1. **查看TensorBoard**: 先启动TensorBoard查看训练曲线
2. **添加调试输出**: 如果需要了解运行时维度，修改代码添加打印
3. **运行小样本测试**: 用少量数据运行模型，观察维度变化
4. **对比CLIP模型**: 用同样的方法分析CLIP模型的维度

## 6. 快速命令参考

```bash
# 查看checkpoint内容
python -c "import torch; ckpt=torch.load('path/to/ckpt.LAST.pth', map_location='cpu'); print(list(ckpt.keys()))"

# 启动TensorBoard
tensorboard --logdir=DATA/output/AirVLN-seq2seq-s/train/TensorBoard/

# 列出所有checkpoint
ls -la DATA/output/AirVLN-seq2seq-s/train/checkpoint/20250922-181721-497138/
```

您的训练数据是完整的，可以很好地用于分析模型结构和训练过程！