# AirVLN BLIP-2 训练流程与架构文档

本文档详细说明了 AirVLN 框架中集成 BLIP-2 模型后的训练流程、数据处理机制以及特征流向。

## 1. 系统架构概览

AirVLN 框架主要由以下部分组成：
- **环境 (AirSim)**: 提供无人机飞行的仿真环境。
- **数据收集 (Data Collection)**: 运行代理在环境中交互，收集图像和轨迹数据，并提取特征保存到 LMDB。
- **训练 (Training)**: 从 LMDB 加载预计算的特征，训练导航策略 (Policy)。
- **模型 (Model)**: 使用 BLIP-2 作为视觉和指令编码器，CMA (Cross-Modal Attention) 作为导航策略。

## 2. BLIP-2 编码器集成

我们实现了三个核心编码器，均基于 `Salesforce/blip2-opt-2.7b` 模型：

### 2.1 BLIP2VisionEncoder (RGB)
- **输入**: 224x224 RGB 图像。
- **核心模型**: BLIP-2 的 Vision Transformer (ViT-g/14)。
- **特征提取**:
  - 提取 ViT 最后一层的输出。
  - 原始特征维度: `[Batch, 257, 1408]` (1个 CLS token + 256个 Patch tokens)。
  - 去除 CLS token 后: `[Batch, 256, 1408]`。
- **特征保存**: 通过 `layer_extract` (Identity层) 暴露给 `train.py` 的 hook，保存 `[Batch, 256, 1408]` 的原始特征到 LMDB。
- **策略输入**:
  - 经过线性投影层 (`projection`) 将维度从 1408 降维到 `output_size` (默认 256)。
  - 最终输出形状: `[Batch, 256, 16, 16]` (空间特征) 或 `[Batch, 256]` (全局特征)。

### 2.2 BLIP2DepthEncoder (Depth)
- **输入**: 256x256 深度图 (处理时调整为 224x224)。
- **处理**: 将单通道深度图复制为 3 通道，模拟 RGB 输入以适配 BLIP-2。
- **特征流向**: 与 RGB 编码器一致，保存原始特征，策略使用投影后的特征。

### 2.3 BLIP2InstructionEncoder (Text)
- **输入**: 文本指令 (Tokenized)。
- **核心模型**: BLIP-2 的 Language Model (OPT-2.7b)。
- **输出**: 提取语言模型最后一层的 hidden states。
- **用途**: 为 CMA 策略提供文本的上下文嵌入。

## 3. 训练流程详解

### 阶段一：数据收集 (Data Collection)
**脚本**: `scripts/collect_ws.sh` -> `src/vlnce_src/train.py (collect_data)`

1.  **环境交互**: 代理在 AirSim 中执行动作，获取 RGB 和 Depth 图像。
2.  **前向传播**:
    - 图像传入 `BLIP2VisionEncoder` 和 `BLIP2DepthEncoder`。
    - 编码器内部 hook 捕获 ViT 输出。
    - 特征通过 `self.layer_extract` (Identity层)。
3.  **特征捕获**:
    - `train.py` 在 `layer_extract` 上注册了外部 hook。
    - Hook 截获 `[Batch, 256, 1408]` 的 Tensor。
4.  **存储**:
    - 特征被保存到 LMDB 数据库中 (键名如 `rgb_features`, `depth_features`)。
    - 原始图像数据通常被丢弃以节省空间 (取决于 `ablate_rgb` 等参数)。

### 阶段二：策略训练 (Training)
**脚本**: `scripts/train_blip2.sh` -> `src/vlnce_src/train.py (train_vlnce)`

1.  **数据加载**:
    - `DDPIWTrajectoryDataset` 从 LMDB 读取数据。
    - `observations` 字典中包含 `rgb_features` 和 `depth_features`，而不是原始图像。
2.  **模型初始化**:
    - 初始化 `CMAPolicy`，进而初始化 `BLIP2VisionEncoder` 等。
    - 设置 `freeze_backbone=True` (通常使用预计算特征时冻结骨干)。
3.  **策略前向传播 (`policy.act` / `forward`)**:
    - 调用 `rgb_encoder(observations)`。
    - **关键逻辑**: 编码器检测到 `observations` 中存在 `rgb_features`。
    - **旁路处理**:
        - 跳过 ViT 模型执行。
        - 直接使用加载的 `rgb_features` (`[Batch, 256, 1408]`)。
        - 通过 `layer_extract` (无操作)。
        - 执行 `projection` 层: `1408 -> 256`。
    - **CMA 融合**:
        - 投影后的视觉特征与指令编码特征在 `CMANet` 中通过 Cross-Modal Attention 进行融合。
        - RNN 更新状态，输出动作概率。
4.  **反向传播**:
    - 计算损失 (如模仿学习损失)。
    - 更新 `projection` 层、CMA 层和 RNN 的参数 (如果骨干被冻结)。

## 4. 关键代码修改点 (本次修复)

为了确保上述流程顺畅，我们对 `Model/encoders/blip2_encoder.py` 进行了以下关键修复：

1.  **添加 `layer_extract`**:
    - 在 `__init__` 中添加 `self.layer_extract = nn.Identity()`。
    - 目的: 为 `train.py` 提供一个稳定的 hook 挂载点，确保能捕获到正确的特征张量。

2.  **支持预计算特征**:
    - 修改 `forward` 方法，优先检查 `observations['rgb_features']`。
    - 如果存在预计算特征，直接使用并跳过繁重的模型推理，确保训练效率和逻辑正确性。

## 5. 注意事项

- **显存占用**: BLIP-2 模型较大，数据收集时显存压力大。训练时由于使用预计算特征，显存主要消耗在特征加载和策略网络上。
- **特征维度**: 确保 LMDB 中保存的特征维度 (`1408`) 与编码器预期的输入维度一致。
- **Hook 机制**: `train.py` 依赖 `layer_extract` 属性，请勿删除或重命名该属性。
