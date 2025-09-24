# AirVLN 模型张量维度详细分析

## 目录
1. [Seq2Seq 模型维度分析](#seq2seq-模型维度分析)
2. [原始 CMA 模型维度分析](#原始-cma-模型维度分析)
3. [CLIP-CMA 模型维度分析](#clip-cma-模型维度分析)
4. [对比总结](#对比总结)

---

## Seq2Seq 模型维度分析

### 输入阶段
```
观测输入:
├── RGB图像: [batch_size, 224, 224, 3]
├── 深度图像: [batch_size, 256, 256, 1]  
└── 指令文本: [batch_size, max_seq_len] (token IDs)
```

### 编码器阶段

#### 1. 指令编码器 (BiLSTM)
```
配置:
├── embedding_size: 50
├── hidden_size: 128
├── bidirectional: False (seq2seq)
└── final_state_only: True

流程:
指令 [batch_size, seq_len] 
→ 嵌入 [batch_size, seq_len, 50]
→ LSTM [batch_size, seq_len, 128]
→ 最终状态 [batch_size, 128]
```

#### 2. RGB编码器 (ResNet50)
```
配置:
├── output_size: 256
└── spatial_output: False (seq2seq)

流程:
RGB [batch_size, 224, 224, 3]
→ ResNet50特征 [batch_size, 2048, 7, 7]
→ 全连接投影 [batch_size, 256]
```

#### 3. 深度编码器 (ResNet50)
```
配置:
├── output_size: 128
└── spatial_output: False (seq2seq)

流程:
深度 [batch_size, 256, 256, 1]
→ ResNet50特征 [batch_size, 2048, 8, 8]
→ 全连接投影 [batch_size, 128]
```

### 融合与输出阶段
```
特征融合:
├── 指令特征: [batch_size, 128]
├── RGB特征: [batch_size, 256]
├── 深度特征: [batch_size, 128]
└── (可选)动作嵌入: [batch_size, 32]

连接后: [batch_size, 128+256+128(+32)] = [batch_size, 512/544]

RNN状态编码器:
输入 [batch_size, 512/544] → LSTM → 输出 [batch_size, 512]
```

---

## 原始 CMA 模型维度分析

### 输入阶段
```
观测输入:
├── RGB图像: [batch_size, 224, 224, 3]
├── 深度图像: [batch_size, 256, 256, 1]
└── 指令文本: [batch_size, max_seq_len] (token IDs)
```

### 编码器阶段

#### 1. 指令编码器 (BiLSTM)
```
配置:
├── embedding_size: 50
├── hidden_size: 128
├── bidirectional: True (CMA)
└── final_state_only: False

流程:
指令 [batch_size, seq_len]
→ 嵌入 [batch_size, seq_len, 50]
→ BiLSTM [batch_size, seq_len, 256] (128*2)
→ 转置 [batch_size, 256, seq_len] (用于attention)
```

#### 2. RGB编码器 (ResNet50)
```
配置:
├── output_size: 256
└── spatial_output: True (CMA)

流程:
RGB [batch_size, 224, 224, 3]
→ ResNet50特征 [batch_size, 2048, 4, 4]
→ 空间嵌入 [batch_size, 64, 4, 4]
→ 连接 [batch_size, 2048+64, 4, 4] = [batch_size, 2112, 4, 4]
```

#### 3. 深度编码器 (ResNet50)
```
配置:
├── output_size: 128
└── spatial_output: True (CMA)

流程:
深度 [batch_size, 256, 256, 1]
→ ResNet50特征 [batch_size, 2048, 8, 8]
→ 空间嵌入 [batch_size, 64, 8, 8]
→ 连接 [batch_size, 2048+64, 8, 8] = [batch_size, 2112, 8, 8]
```

### CMA注意力机制
```
第一阶段RNN输入:
├── RGB线性投影: [batch_size, 2112, 16] → AdaptiveAvgPool1d → [batch_size, 256]
├── 深度线性投影: [batch_size, 2112, 64] → Flatten → [batch_size, 128]
└── 动作嵌入: [batch_size, 32]

连接: [batch_size, 256+128+32] = [batch_size, 416]
→ 第一RNN: [batch_size, 416] → [batch_size, 512]

Cross-Modal Attention:
1. 文本-状态注意力:
   ├── state_q: [batch_size, 512] → [batch_size, 256]
   ├── text_k: [batch_size, 256, seq_len] → [batch_size, 256, seq_len]
   └── 注意力结果: [batch_size, 256]

2. 文本-视觉注意力:
   ├── text_q: [batch_size, 256] → [batch_size, 256]
   ├── rgb_kv: [batch_size, 2112, 16] → split → k:[batch_size, 256, 16], v:[batch_size, 256, 16]
   ├── depth_kv: [batch_size, 2112, 64] → split → k:[batch_size, 256, 64], v:[batch_size, 128, 64]
   ├── RGB注意力: [batch_size, 256]
   └── 深度注意力: [batch_size, 128]

最终融合:
[batch_size, 512+256+256+128+32] = [batch_size, 1184]
→ 压缩: [batch_size, 512]
→ 第二RNN: [batch_size, 512]
```

---

## CLIP-CMA 模型维度分析

### 输入阶段
```
观测输入:
├── RGB图像: [batch_size, 224, 224, 3]
├── 深度图像: [batch_size, 256, 256, 1]
└── 指令文本: [batch_size, max_seq_len] (token IDs)
```

### CLIP编码器阶段

#### 1. CLIP指令编码器
```
配置:
├── CLIP模型: openai/clip-vit-base-patch32
├── hidden_size: 512 (CLIP text hidden)
└── final_state_only: False

流程:
指令 [batch_size, seq_len]
→ CLIP Text Model
→ 序列输出 [batch_size, seq_len, 512]
→ 转置 [batch_size, 512, seq_len] (用于attention)
```

#### 2. CLIP RGB编码器
```
配置:
├── CLIP模型: openai/clip-vit-base-patch32
├── output_size: 256 (投影后)
├── spatial_output: True
└── spatial_size: 4x4

流程:
RGB [batch_size, 224, 224, 3]
→ CLIP Vision Model [batch_size, 768]
→ 空间投影 [batch_size, 256*4*4] → reshape [batch_size, 256, 4, 4]
→ 空间嵌入 [batch_size, 64, 4, 4]
→ 连接 [batch_size, 256+64, 4, 4] = [batch_size, 320, 4, 4]
```

#### 3. CLIP深度编码器
```
配置:
├── CLIP模型: openai/clip-vit-base-patch32 (共享视觉模型)
├── output_size: 128 (投影后)
├── spatial_output: True
└── spatial_size: 8x8

流程:
深度 [batch_size, 256, 256, 1]
→ 单通道转3通道 [batch_size, 3, 224, 224]
→ CLIP Vision Model [batch_size, 768]
→ 空间投影 [batch_size, 128*8*8] → reshape [batch_size, 128, 8, 8]
→ 空间嵌入 [batch_size, 64, 8, 8]
→ 连接 [batch_size, 128+64, 8, 8] = [batch_size, 192, 8, 8]
```

### CLIP-CMA注意力机制
```
第一阶段RNN输入:
├── RGB线性投影: [batch_size, 320, 16] → AdaptiveAvgPool1d → [batch_size, 256]
├── 深度线性投影: [batch_size, 192, 64] → Flatten → [batch_size, 128]
└── 动作嵌入: [batch_size, 32]

连接: [batch_size, 256+128+32] = [batch_size, 416]
→ 第一RNN: [batch_size, 416] → [batch_size, 512]

Cross-Modal Attention:
1. 文本-状态注意力:
   ├── state_q: [batch_size, 512] → [batch_size, 256]
   ├── text_k: [batch_size, 512, seq_len] → [batch_size, 256, seq_len]
   └── 注意力结果: [batch_size, 512] (CLIP text 维度)

2. 文本-视觉注意力:
   ├── text_q: [batch_size, 512] → [batch_size, 256]
   ├── rgb_kv: [batch_size, 320, 16] → split → k:[batch_size, 256, 16], v:[batch_size, 256, 16]
   ├── depth_kv: [batch_size, 192, 64] → split → k:[batch_size, 256, 64], v:[batch_size, 128, 64]
   ├── RGB注意力: [batch_size, 256]
   └── 深度注意力: [batch_size, 128]

最终融合:
[batch_size, 512+512+256+128+32] = [batch_size, 1440]
→ 压缩: [batch_size, 512]
→ 第二RNN: [batch_size, 512]
```

---

## 对比总结

### 参数量对比
| 模型 | 指令编码器 | RGB编码器 | 深度编码器 | 总计 |
|------|------------|-----------|------------|------|
| Seq2Seq | BiLSTM (~1M) | ResNet50 (~25M) | ResNet50 (~25M) | ~51M |
| 原始CMA | BiLSTM (~1M) | ResNet50 (~25M) | ResNet50 (~25M) | ~51M |
| CLIP-CMA | CLIP Text (~37M) | CLIP Vision (~85M, 共享) | CLIP Vision (共享) | ~124M |

### 特征维度对比
| 模型 | 指令特征 | RGB特征 | 深度特征 | 最终输出 |
|------|----------|---------|----------|----------|
| Seq2Seq | [B, 128] | [B, 256] | [B, 128] | [B, 512] |
| 原始CMA | [B, 256, L] | [B, 2112, 4, 4] | [B, 2112, 8, 8] | [B, 512] |
| CLIP-CMA | [B, 512, L] | [B, 320, 4, 4] | [B, 192, 8, 8] | [B, 512] |

### 关键差异
1. **语义对齐**: CLIP模型预训练的视觉-语言对齐能力
2. **特征维度**: CLIP文本特征维度更大 (512 vs 256)
3. **参数共享**: RGB和深度编码器共享CLIP视觉模型
4. **内存使用**: CLIP模型更大但可以选择冻结backbone

### 预期优势
1. **更强的跨模态理解**: CLIP预训练的多模态对齐
2. **更好的泛化能力**: 预训练模型的知识迁移  
3. **统一的视觉表示**: RGB和深度使用同一模型
4. **更丰富的语义特征**: 更大的特征维度

---

**注释**: 
- B = batch_size, L = sequence_length
- 所有维度为理论计算，实际运行时可能因具体配置而略有差异
- CLIP模型可选择不同尺寸 (base/large)，对应维度会相应调整