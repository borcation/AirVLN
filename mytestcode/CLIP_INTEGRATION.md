# CLIP编码器集成说明

## 概述
我们已经成功将CLIP (Contrastive Language-Image Pre-training) 编码器集成到AirVLN项目中，以替换原始的ResNet图像编码器和BiLSTM文本编码器。CLIP编码器能够更好地处理图像和文本之间的语义对齐，从而提升模型的导航性能。

## 主要改进

### 1. 图像编码器 (CLIPVisionEncoder)
- **替换**: ResNet50 → CLIP Vision Transformer
- **特征维度**: 2048 → 768 (ViT-Base) 或 1024 (ViT-Large)
- **优势**: 预训练的视觉-语言对齐能力，更好的场景理解

### 2. 深度编码器 (CLIPDepthEncoder) 🆕
- **替换**: ResNet50 Depth → CLIP Vision Transformer
- **特征维度**: 2048 → 768 (ViT-Base)
- **预处理**: 单通道深度图转换为3通道RGB格式输入CLIP
- **优势**: 统一的视觉表示，更好的深度语义理解

### 3. 文本编码器 (CLIPInstructionEncoder)  
- **替换**: BiLSTM → CLIP Text Transformer
- **特征维度**: 256 (BiLSTM hidden*2) → 512 (CLIP-Base)
- **优势**: 更强的文本理解和与视觉特征的语义对齐

### 4. 架构兼容性
- 保持CMA (Cross-Modal Attention) 机制不变
- 输出维度适配原始架构要求
- 支持空间特征提取用于attention机制
- RGB和深度图像共享CLIP视觉模型，节省内存

## 使用方法

### 训练
```bash
# 完全使用CLIP编码器训练（RGB + 深度 + 文本）
cd AirVLN
bash scripts/train_clip.sh

# 或者手动设置参数
python -u -m src.vlnce_src.run \
    --run_type train \
    --policy_type cma \
    --use_clip_encoders \
    --use_clip_depth_encoder \
    --clip_model_name openai/clip-vit-base-patch32 \
    --freeze_clip_backbone \
    --batchSize 4 \
    --epochs 5 \
    --name AirVLN-cma-clip-full

# 仅RGB和文本使用CLIP，深度保持ResNet
python -u -m src.vlnce_src.run \
    --run_type train \
    --policy_type cma \
    --use_clip_encoders \
    --clip_model_name openai/clip-vit-base-patch32 \
    --freeze_clip_backbone \
    --batchSize 4 \
    --epochs 5 \
    --name AirVLN-cma-clip-rgb-text
```

### 评估
```bash
# 评估CLIP模型
bash scripts/eval_clip.sh

# 或者手动设置参数
python -u -m src.vlnce_src.run \
    --run_type eval \
    --policy_type cma \
    --use_clip_encoders \
    --clip_model_name openai/clip-vit-base-patch32 \
    --EVAL_CKPT_PATH_DIR path/to/clip/checkpoints
```

## 配置参数

### 新增参数
- `--use_clip_encoders`: 启用CLIP编码器（替换ResNet RGB + BiLSTM）
- `--use_clip_depth_encoder`: 同时对深度图像使用CLIP编码器
- `--clip_model_name`: CLIP模型名称（默认: openai/clip-vit-base-patch32）
- `--freeze_clip_backbone`: 冻结CLIP backbone参数（推荐用于快速训练）

### 支持的CLIP模型
- `openai/clip-vit-base-patch32` (推荐)
- `openai/clip-vit-base-patch16`
- `openai/clip-vit-large-patch14`

## 技术细节

### 维度映射
| 组件 | 原始维度 | CLIP维度 | 说明 |
|------|----------|----------|------|
| RGB特征 | 256 | 256 | 通过投影层适配 |
| RGB空间特征 | (2048+64, 4, 4) | (256+64, 4, 4) | 保持空间attention兼容性 |
| 深度特征 | 128 | 128 | 通过投影层适配 |
| 深度空间特征 | (2048+64, 8, 8) | (128+64, 8, 8) | 保持原始深度编码器维度 |
| 文本特征 | 256 | 512 | CLIP text hidden size |
| 总输出 | 1152 | 1408 | CMA最终特征维度 |

### 深度图像处理
- **输入格式**: 单通道深度图 [H, W, 1]
- **预处理**: 转换为3通道RGB格式 [3, 224, 224]
- **归一化**: 深度值归一化到 [0, 1] 范围
- **共享模型**: RGB和深度图像使用同一个CLIP视觉模型

### 内存使用
- CLIP模型参数量: ~151M (ViT-Base)
- 建议GPU内存: ≥12GB（完全CLIP） / ≥8GB（部分CLIP）
- 使用`--freeze_clip_backbone`可减少内存使用
- RGB和深度共享CLIP视觉模型，节省约85M参数

## 训练配置建议

### 配置1: 渐进式集成
```bash
# 步骤1: RGB + 文本使用CLIP
--use_clip_encoders

# 步骤2: 添加深度CLIP编码器  
--use_clip_encoders --use_clip_depth_encoder
```

### 配置2: 内存优化
```bash
# 冻结backbone减少内存
--use_clip_encoders --use_clip_depth_encoder --freeze_clip_backbone --batchSize 4

# 精细调优（需要更多内存）
--use_clip_encoders --use_clip_depth_encoder --batchSize 2 --lr 0.0001
```

## 实验结果对比

### 预期改进
1. **语义理解**: CLIP的预训练优势提升指令理解
2. **视觉特征**: 更丰富的场景语义表示
3. **多模态对齐**: 图像-文本特征更好的语义一致性

### 建议实验设置
- 初始学习率: 0.00025 (与原始设置相同)
- Batch size: 4-8 (根据GPU内存调整)
- 冻结backbone进行快速实验，然后fine-tune获得最佳性能

## 故障排除

### 常见问题
1. **内存不足**: 使用`--freeze_clip_backbone`或减小batch size
2. **模型下载失败**: 确保网络连接，或使用本地模型路径
3. **维度不匹配**: 检查args配置，确保所有编码器参数一致

### 依赖要求
```bash
# 更新transformers版本
pip install transformers==4.21.3
pip install accelerate>=0.20.0
```

## 未来改进方向

1. **更大模型**: 实验CLIP-Large提升性能
2. **Fine-tuning策略**: 渐进式解冻训练
3. **多尺度特征**: 利用CLIP的多层特征
4. **领域适应**: 针对无人机场景的CLIP微调

---

**注意**: 首次使用时会自动下载CLIP预训练模型（~500MB），请确保网络连接稳定。