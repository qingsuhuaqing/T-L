# Time-LLM 深度技术解析 (Deep Technical Analysis)

> **从"怎么跑"到"怎么懂"** —— Time-LLM 核心机制完全剖析

---

## 一、代码库拓扑图 (Project Topology)

### 1.1 项目结构树

```
Time-LLM/
├── models/                          # 核心模型定义
│   ├── __init__.py
│   ├── TimeLLM.py                  # ★★★ Time-LLM 主模型（核心）
│   ├── Autoformer.py               # 基线模型：Autoformer
│   └── DLinear.py                  # 基线模型：DLinear
│
├── layers/                          # 神经网络层组件
│   ├── __init__.py
│   ├── Embed.py                    # ★★★ PatchEmbedding（核心）
│   ├── StandardNorm.py             # ★★★ 实例归一化层（核心）
│   ├── Transformer_EncDec.py       # Transformer 编码器/解码器
│   ├── Autoformer_EncDec.py        # Autoformer 编码器/解码器
│   ├── SelfAttention_Family.py     # 注意力机制族
│   ├── AutoCorrelation.py          # 自相关机制
│   └── Conv_Blocks.py              # 卷积块
│
├── data_provider/                   # 数据加载与处理
│   ├── __init__.py
│   ├── data_factory.py             # ★ 数据集工厂（路由器）
│   ├── data_loader.py              # ★★ 数据加载器（ETT/Weather/ECL/Traffic）
│   └── m4.py                       # M4 竞赛数据集加载器
│
├── data_provider_pretrain/          # 预训练数据加载（迁移学习）
│   ├── __init__.py
│   ├── data_factory.py
│   └── data_loader.py
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   ├── tools.py                    # ★ 训练工具（EarlyStopping, vali, load_content）
│   ├── metrics.py                  # ★ 评估指标（MAE, MSE, RMSE, MAPE, MSPE）
│   ├── losses.py                   # 损失函数
│   ├── timefeatures.py             # 时间特征编码
│   └── m4_summary.py               # M4 评估汇总
│
├── dataset/                         # 数据集存放目录
│   ├── prompt_bank/                # ★★ 领域描述提示词库
│   │   ├── ETT.txt                 # ETT 数据集描述
│   │   ├── Weather.txt             # Weather 数据集描述
│   │   ├── ECL.txt                 # Electricity 数据集描述
│   │   └── m4.txt                  # M4 数据集描述
│   ├── ETT-small/                  # ETT 数据集（需下载）
│   ├── weather/                    # Weather 数据集（需下载）
│   ├── electricity/                # Electricity 数据集（需下载）
│   └── traffic/                    # Traffic 数据集（需下载）
│
├── scripts/                         # 训练脚本
│   ├── TimeLLM_ETTh1.sh
│   ├── TimeLLM_ETTh2.sh
│   ├── TimeLLM_ETTm1.sh
│   ├── TimeLLM_ETTm2.sh
│   ├── TimeLLM_Weather.sh
│   ├── TimeLLM_ECL.sh
│   ├── TimeLLM_Traffic.sh
│   └── TimeLLM_M4.sh
│
├── checkpoints/                     # 模型检查点保存目录（训练时自动创建）
│
├── run_main.py                      # ★★★ 主训练入口（长期预测）
├── run_m4.py                        # M4 短期预测入口
├── run_pretrain.py                  # 预训练/迁移学习入口
│
├── ds_config_zero2.json             # DeepSpeed ZeRO-2 配置
├── requirements.txt                 # Python 依赖列表
├── README.md                        # 项目说明
├── CLAUDE.md                        # Claude Code 项目指南
└── LICENSE                          # 许可证
```

---

### 1.2 核心模块职责详解

#### 📦 `models/` - 模型定义层

| 文件 | 职责 | 核心类/函数 |
|------|------|-----------|
| **TimeLLM.py** | Time-LLM 主模型实现 | `Model`（主模型）、`ReprogrammingLayer`（重编程层）、`FlattenHead`（输出投影） |
| Autoformer.py | Autoformer 基线模型 | `Model` |
| DLinear.py | DLinear 基线模型 | `Model` |

**TimeLLM.py 核心组件：**
- **LLM 加载模块**（第 43-154 行）：支持 LLAMA、GPT-2、BERT 三种基座模型
- **PatchEmbedding**（第 173-174 行）：将时间序列切分为 Patch 并嵌入
- **Mapping Layer**（第 179 行）：将 LLM 词嵌入空间映射为可学习的 Token 空间
- **ReprogrammingLayer**（第 181 行 + 第 267-305 行）：跨模态对齐层，将时序 Patch 映射到 LLM 嵌入空间
- **Normalize 层**（第 192 行）：实例归一化，保存统计量用于反归一化
- **FlattenHead**（第 15-27 行）：输出投影层，将 LLM 输出映射为预测序列

---

#### 🧱 `layers/` - 神经网络层组件

| 文件 | 职责 | 核心类 |
|------|------|--------|
| **Embed.py** | 嵌入层（Token/Positional/Temporal/Patch） | `PatchEmbedding`（核心）、`TokenEmbedding`、`PositionalEmbedding` |
| **StandardNorm.py** | 实例归一化（RevIN） | `Normalize`（支持 norm/denorm 双向操作） |
| Transformer_EncDec.py | Transformer 编码器/解码器 | `Encoder`, `Decoder`, `EncoderLayer`, `DecoderLayer` |
| SelfAttention_Family.py | 注意力机制 | `FullAttention`, `ProbAttention`, `AttentionLayer` |

**PatchEmbedding 核心机制（Embed.py 第 160-186 行）：**
```python
class PatchEmbedding(nn.Module):
    def forward(self, x):
        # 1. Padding：复制最后一个值进行填充
        x = self.padding_patch_layer(x)

        # 2. Unfold：滑动窗口切分 Patch
        # x.shape: [B, N, T] -> [B, N, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 3. Reshape：展平 Batch 和变量维度
        # [B, N, num_patches, patch_len] -> [B*N, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # 4. TokenEmbedding：使用 1D 卷积将 Patch 映射到 d_model 维度
        # [B*N, num_patches, patch_len] -> [B*N, num_patches, d_model]
        x = self.value_embedding(x)

        return self.dropout(x), n_vars
```

---

#### 📊 `data_provider/` - 数据管道层

| 文件 | 职责 | 核心功能 |
|------|------|---------|
| **data_factory.py** | 数据集路由器 | `data_provider()` 函数：根据 `args.data` 选择对应的 Dataset 类 |
| **data_loader.py** | 数据集加载器 | `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom` |
| m4.py | M4 竞赛数据加载 | `M4Dataset`, `M4Meta` |

**数据集切分策略（以 ETTh1 为例）：**
- **Train**：前 12 个月（8640 小时）
- **Validation**：中间 4 个月（2880 小时）
- **Test**：最后 4 个月（2880 小时）

**数据加载流程：**
1. 读取 CSV 文件
2. 选择特征列（M：多变量，S：单变量，MS：多变量预测单变量）
3. StandardScaler 标准化（使用训练集统计量）
4. 时间特征编码（月/日/星期/小时）
5. 返回滑动窗口样本：`(seq_x, seq_y, seq_x_mark, seq_y_mark)`

---

#### 🛠️ `utils/` - 工具函数层

| 文件 | 职责 | 核心函数 |
|------|------|---------|
| **tools.py** | 训练工具 | `EarlyStopping`（早停）、`vali()`（验证）、`adjust_learning_rate()`、`load_content()`（加载 Prompt） |
| **metrics.py** | 评估指标 | `MAE()`, `MSE()`, `RMSE()`, `MAPE()`, `MSPE()`, `metric()`（一次性计算全部） |
| losses.py | 损失函数 | `mape_loss`, `mase_loss`, `smape_loss`（M4 使用） |
| timefeatures.py | 时间特征 | `time_features()` 提取周期性特征 |

---

#### 📝 `dataset/prompt_bank/` - 提示词库

存储每个数据集的领域描述文本，用于构建动态 Prompt。

**示例（ETT.txt）：**
```
The Electricity Transformer Temperature (ETT) is a crucial indicator in the
electric power long-term deployment.
```

在 `TimeLLM.py` 第 220 行构建完整 Prompt：
```python
prompt_ = (
    f"<|start_prompt|>Dataset description: {self.description}"
    f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
    "Input statistics: "
    f"min value {min_values_str}, "
    f"max value {max_values_str}, "
    f"median value {median_values_str}, "
    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
)
```

---

## 二、核心数据流机制 (Data Flow & Pipeline)

### 2.1 端到端数据流概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Time-LLM 数据流全景图                              │
└─────────────────────────────────────────────────────────────────────────┘

原始时序数据 (CSV)
    ↓
[1] 数据加载 & 预处理 (data_loader.py)
    ├─ StandardScaler 标准化
    ├─ 时间特征编码 (月/日/星期/小时)
    └─ 滑动窗口切片
    ↓
输入张量: x_enc [Batch, SeqLen, N_vars]
    ↓
[2] 实例归一化 (Normalize Layer)
    ├─ 计算均值/方差
    └─ Z-score 标准化
    ↓
归一化数据: x_enc [Batch, SeqLen, N_vars]
    ↓
[3] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┃  Prompt 构建 (统计特征提取)          ┃
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─ 计算 min, max, median
    ├─ 计算趋势 (diff().sum())
    ├─ FFT 自相关 -> Top-5 Lags
    └─ 拼接领域描述 + 统计信息
    ↓
文本 Prompt → Tokenizer → prompt_embeddings [Batch, PromptLen, llm_dim]
    ↓
[4] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┃  Patching (时序分块嵌入)             ┃
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─ Unfold: 滑动窗口切分 Patch
    │   x_enc [B, N, SeqLen] → [B, N, num_patches, patch_len]
    ├─ Reshape: 展平 Batch 和变量
    │   [B, N, num_patches, patch_len] → [B*N, num_patches, patch_len]
    └─ TokenEmbedding (1D Conv): 投影到 d_model
        [B*N, num_patches, patch_len] → [B*N, num_patches, d_model]
    ↓
Patch Embeddings: enc_out [B*N, num_patches, d_model]
    ↓
[5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┃  Reprogramming (跨模态对齐)          ┃
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─ Source Embeddings: LLM 词嵌入 → Mapping Layer
    │   word_embeddings [vocab_size, llm_dim] → [num_tokens, llm_dim]
    ├─ Cross-Attention (Query: Patch, Key/Value: Source)
    │   Q = Linear(enc_out) [B*N, num_patches, d_keys*n_heads]
    │   K = Linear(source_embeddings) [num_tokens, d_keys*n_heads]
    │   V = Linear(source_embeddings) [num_tokens, d_keys*n_heads]
    │   Attention = softmax(Q @ K^T / sqrt(d_keys)) @ V
    └─ Output Projection: 映射到 llm_dim
        reprogrammed [B*N, num_patches, llm_dim]
    ↓
重编程后的 Embeddings: enc_out [B*N, num_patches, llm_dim]
    ↓
[6] 拼接 Prompt + Patch Embeddings
    llama_enc_out = Concat([prompt_embeddings, enc_out], dim=1)
    Shape: [B*N, PromptLen + num_patches, llm_dim]
    ↓
[7] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┃  LLM Forward (冻结的 LLM Backbone)   ┃
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    └─ GPT2/LLAMA/BERT 前向传播（参数冻结）
    ↓
LLM 输出: dec_out [B*N, PromptLen + num_patches, llm_dim]
    ↓
[8] 截取前 d_ff 维度 + 提取 Patch 部分
    dec_out = dec_out[:, :, :d_ff]  # 取前 d_ff 维
    dec_out = Reshape([B, N, total_len, d_ff])
    dec_out = dec_out[:, :, :, -num_patches:]  # 提取 Patch 对应的输出
    ↓
[9] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┃  FlattenHead (输出投影)              ┃
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─ Flatten: [B, N, d_ff, num_patches] → [B, N, d_ff * num_patches]
    └─ Linear: [B, N, d_ff * num_patches] → [B, N, pred_len]
    ↓
预测结果 (归一化空间): dec_out [Batch, pred_len, N_vars]
    ↓
[10] 反归一化 (Denormalize)
    dec_out = dec_out * stdev + mean
    ↓
最终预测: dec_out [Batch, pred_len, N_vars]
```

---

### 2.2 关键机制深度解析

#### 🔹 机制 1: Patching（时序分块）

**核心思想：** 将长时间序列切分为固定长度的 Patch，类似 Vision Transformer (ViT) 的图像分块。

**代码位置：** `layers/Embed.py` 第 177-185 行

**详细流程：**

1. **Padding（第 180 行）**
   ```python
   x = self.padding_patch_layer(x)  # ReplicationPad1d
   # 复制最后一个时间步的值，填充 stride 长度
   # 目的：确保序列长度能被 stride 整除
   ```

2. **Unfold（第 181 行）**
   ```python
   x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
   # 滑动窗口切分
   # 输入: [B, N_vars, SeqLen + stride]
   # 输出: [B, N_vars, num_patches, patch_len]
   # num_patches = (SeqLen - patch_len) / stride + 2
   ```

   **示例：**
   ```
   SeqLen = 96, patch_len = 16, stride = 8
   num_patches = (96 - 16) / 8 + 2 = 12

   原始序列: [x0, x1, x2, ..., x95]
   Patch 0: [x0, x1, ..., x15]
   Patch 1: [x8, x9, ..., x23]  (stride=8, 重叠 50%)
   Patch 2: [x16, x17, ..., x31]
   ...
   Patch 11: [x88, x89, ..., x103] (padding 部分)
   ```

3. **Reshape（第 182 行）**
   ```python
   x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
   # 展平 Batch 和变量维度
   # [B, N_vars, num_patches, patch_len] → [B*N_vars, num_patches, patch_len]
   # 目的：将每个变量的每个 Patch 当作独立样本处理
   ```

4. **TokenEmbedding（第 184 行）**
   ```python
   x = self.value_embedding(x)  # 1D Conv: in_channels=patch_len, out_channels=d_model
   # [B*N_vars, num_patches, patch_len] → [B*N_vars, num_patches, d_model]
   ```

**为什么使用 Patching？**
- **降低计算复杂度：** Transformer 的复杂度是 O(L²)，Patching 将序列长度从 L 降为 num_patches
- **局部模式捕捉：** 每个 Patch 保留局部时序依赖
- **与 LLM Token 对齐：** Patch 类比文本中的 Token，更符合 LLM 的输入形式

---

#### 🔹 机制 2: Reprogramming（重编程/跨模态对齐）

**核心思想：** 通过 **Cross-Attention** 机制，将时序 Patch Embeddings（来自时序域）映射到 LLM 的词嵌入空间（文本域），实现跨模态对齐。

**代码位置：** `models/TimeLLM.py` 第 267-305 行 `ReprogrammingLayer`

**详细流程：**

1. **Source Embeddings 构建（第 237 行）**
   ```python
   # 获取 LLM 的词嵌入矩阵
   self.word_embeddings = self.llm_model.get_input_embeddings().weight
   # Shape: [vocab_size, llm_dim]，例如 GPT2: [50257, 768]

   # Mapping Layer: 将词表空间压缩为可学习的 Token 空间
   source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
   # [vocab_size, llm_dim] → [llm_dim, vocab_size] → Linear → [llm_dim, num_tokens] → [num_tokens, llm_dim]
   # num_tokens = 1000 (可学习的虚拟词表大小)
   ```

2. **Cross-Attention 对齐（第 280-305 行）**
   ```python
   # Query: 来自 Patch Embeddings（时序域）
   Q = self.query_projection(target_embedding)  # [B*N, num_patches, d_keys * n_heads]

   # Key & Value: 来自 Source Embeddings（LLM 词嵌入空间）
   K = self.key_projection(source_embedding)    # [num_tokens, d_keys * n_heads]
   V = self.value_projection(value_embedding)   # [num_tokens, d_keys * n_heads]

   # 计算注意力分数
   scores = torch.einsum("blhe,she->bhls", Q, K)  # [B*N, n_heads, num_patches, num_tokens]
   # l: num_patches, s: num_tokens, h: n_heads, e: d_keys

   # Softmax 归一化
   A = softmax(scores / sqrt(d_keys), dim=-1)  # [B*N, n_heads, num_patches, num_tokens]

   # 加权聚合 Value
   out = torch.einsum("bhls,she->blhe", A, V)  # [B*N, num_patches, n_heads, d_keys]

   # 输出投影到 llm_dim
   out = self.out_projection(out.reshape(B, L, -1))  # [B*N, num_patches, llm_dim]
   ```

**物理意义：**
- **Query（时序域）：** "我是一个时序 Patch，我想找到文本词表中与我语义最接近的词"
- **Key（文本域）：** "我是 LLM 词表中的一个词，我代表某种语义"
- **Attention 权重：** 度量时序 Patch 与文本词之间的语义相似度
- **Output：** 时序 Patch 在 LLM 嵌入空间中的表示（融合了文本语义）

**为什么需要 Reprogramming？**
- **模态鸿沟：** 时序数据（连续数值）与文本数据（离散 Token）分布差异巨大
- **冻结 LLM：** LLM 参数冻结，无法直接适配时序数据，需要通过 Reprogramming 层"翻译"
- **知识迁移：** 利用 LLM 的预训练知识（语义理解能力）帮助时序建模

---

### 2.3 数据形状变化全流程

以 **ETTh1 数据集 + GPT-2 模型** 为例：

| 阶段 | 张量名称 | Shape | 说明 |
|------|---------|-------|------|
| **输入** | x_enc | `[32, 96, 7]` | Batch=32, SeqLen=96, N_vars=7 |
| **归一化后** | x_enc | `[32, 96, 7]` | 实例归一化（减均值除方差） |
| **Prompt 嵌入** | prompt_embeddings | `[32, 128, 768]` | Prompt 长度约 128 Token，GPT2 隐藏层 768 |
| **Patching 后** | enc_out | `[224, 12, 16]` | B*N=32*7=224, num_patches=12, d_model=16 |
| **Reprogramming 后** | enc_out | `[224, 12, 768]` | 映射到 llm_dim=768 |
| **拼接输入** | llama_enc_out | `[224, 140, 768]` | Concat Prompt(128) + Patches(12) = 140 |
| **LLM 输出** | dec_out | `[224, 140, 768]` | GPT2 前向传播 |
| **截取 + Reshape** | dec_out | `[32, 7, 32, 12]` | 取前 d_ff=32 维，Reshape 回 [B, N, d_ff, num_patches] |
| **FlattenHead 输入** | dec_out | `[32, 7, 384]` | Flatten 后两维：32*12=384 |
| **FlattenHead 输出** | dec_out | `[32, 7, 96]` | Linear(384 → pred_len=96) |
| **反归一化后** | dec_out | `[32, 96, 7]` | 最终预测：Batch=32, PredLen=96, N_vars=7 |

**关键参数计算：**
```python
# Patch 数量
num_patches = (seq_len - patch_len) / stride + 2
            = (96 - 16) / 8 + 2 = 12

# FlattenHead 输入维度
head_nf = d_ff * num_patches = 32 * 12 = 384

# Prompt Token 长度（动态，取决于统计信息字符串长度）
prompt_len ≈ 128 (tokenizer 自动 padding/truncation)
```

---

## 三、模型架构解析 (Model Architecture)

### 3.1 冻结 (Frozen) vs 可训练 (Trainable) 参数

#### ❄️ 冻结部分（参数不更新）

| 组件 | 代码位置 | 参数量（以 GPT-2 为例） | 说明 |
|------|---------|----------------------|------|
| **LLM Backbone** | `models/TimeLLM.py` 第 163-164 行 | **117M** | GPT-2 全部参数冻结 |

```python
# 冻结 LLM 参数
for param in self.llm_model.parameters():
    param.requires_grad = False
```

**为什么冻结 LLM？**
1. **参数效率：** LLM 参数量巨大（GPT-2: 117M，LLAMA-7B: 7B），全量微调需要大量显存
2. **知识保留：** 冻结参数保留预训练知识，避免在小数据集上过拟合
3. **计算效率：** 反向传播时不计算 LLM 梯度，大幅降低显存和计算开销

---

#### 🔥 可训练部分（参数更新）

| 组件 | 代码位置 | 形状 | 参数量（示例） | 说明 |
|------|---------|------|--------------|------|
| **PatchEmbedding** | `layers/Embed.py` 第 169 行 | `Conv1d(16, 16, 3)` | ~800 | 将 Patch 嵌入到 d_model |
| **Mapping Layer** | `models/TimeLLM.py` 第 179 行 | `Linear(50257, 1000)` | **50.3M** | 词表映射（GPT-2 词表 50257） |
| **Reprogramming Layer** | `models/TimeLLM.py` 第 181 行 | 4 个 Linear 层 | ~6M | Cross-Attention 权重 |
| **Output Projection (FlattenHead)** | `models/TimeLLM.py` 第 187 行 | `Linear(384, 96)` | ~37K | 输出投影到预测长度 |
| **Normalize Layer** | `models/TimeLLM.py` 第 192 行 | 无参数（仅统计量） | 0 | 实例归一化（非可学习） |

**总可训练参数量：** 约 **56-60M**（取决于 d_model, d_ff, n_heads 配置）

**训练时参数更新：**
```python
# run_main.py 第 148-150 行
trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

# 优化器只优化可训练参数
optimizer = optim.Adam(trained_parameters, lr=args.learning_rate)
```

---

### 3.2 Checkpoint 内容详解

**保存位置：** `checkpoints/{setting}-{model_comment}/checkpoint`

**保存代码：** `utils/tools.py` 第 79-83 行
```python
def save_checkpoint(self, val_loss, model, path):
    if self.accelerator is not None:
        model = self.accelerator.unwrap_model(model)  # 解包 DDP/DeepSpeed 包装
        torch.save(model.state_dict(), path + '/' + 'checkpoint')
```

**Checkpoint 包含的参数：**
```python
checkpoint = {
    # 1. PatchEmbedding
    'patch_embedding.value_embedding.tokenConv.weight': [16, 16, 3],

    # 2. Mapping Layer
    'mapping_layer.weight': [1000, 50257],  # ★ 最大参数块
    'mapping_layer.bias': [1000],

    # 3. Reprogramming Layer
    'reprogramming_layer.query_projection.weight': [d_keys * n_heads, d_model],
    'reprogramming_layer.key_projection.weight': [d_keys * n_heads, llm_dim],
    'reprogramming_layer.value_projection.weight': [d_keys * n_heads, llm_dim],
    'reprogramming_layer.out_projection.weight': [llm_dim, d_keys * n_heads],

    # 4. FlattenHead (Output Projection)
    'output_projection.linear.weight': [pred_len, d_ff * num_patches],
    'output_projection.linear.bias': [pred_len],

    # 注意：LLM 参数不在 Checkpoint 中（因为被冻结）
}
```

**Checkpoint 文件大小：**
- **GPT-2 + d_model=16 + d_ff=32：** 约 **200-250 MB**
- **LLAMA-7B + d_model=32 + d_ff=128：** 约 **300-400 MB**

**加载 Checkpoint 推理：**
```python
# 1. 初始化模型
model = TimeLLM.Model(args).float()

# 2. 加载权重
checkpoint_path = './checkpoints/long_term_forecast_ETTh1_96_96_.../checkpoint'
model.load_state_dict(torch.load(checkpoint_path))

# 3. 推理模式
model.eval()
with torch.no_grad():
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
```

---

## 四、产出与评估 (Outputs & Inference)

### 4.1 训练产出文件

#### 📁 `checkpoints/` 目录结构

```
checkpoints/
└── long_term_forecast_ETTh1_96_96_TimeLLM_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Exp_0-TimeLLM-GPT2-LowMem/
    ├── checkpoint              # ★ 模型权重文件（最佳验证集模型）
    └── [已删除] checkpoint.tmp # 训练结束后自动删除
```

**命名规则：**
```
{task_name}_{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_
dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_{des}_{itr}-{model_comment}
```

**示例：**
```
long_term_forecast_ETTh1_96_96_TimeLLM_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Exp_0-TimeLLM-GPT2-LowMem
```

---

### 4.2 推理流程

#### 🔹 单批次推理示例

```python
import torch
from models import TimeLLM
import argparse

# 1. 配置参数（与训练时保持一致）
args = argparse.Namespace(
    llm_model='GPT2',
    llm_dim=768,
    llm_layers=6,
    d_model=16,
    d_ff=32,
    n_heads=8,
    dropout=0.1,
    seq_len=96,
    pred_len=96,
    patch_len=16,
    stride=8,
    enc_in=7,
    task_name='long_term_forecast',
    prompt_domain=1,
    content='The Electricity Transformer Temperature (ETT) is a crucial indicator...'
)

# 2. 初始化模型
model = TimeLLM.Model(args).float()

# 3. 加载 Checkpoint
model.load_state_dict(torch.load('checkpoints/.../checkpoint'))
model.eval()

# 4. 准备输入数据
batch_x = torch.randn(1, 96, 7)        # [Batch, SeqLen, N_vars]
batch_x_mark = torch.randn(1, 96, 4)   # [Batch, SeqLen, TimeFeatures]
dec_inp = torch.zeros(1, 96, 7)        # [Batch, PredLen, N_vars] (占位符)
batch_y_mark = torch.randn(1, 96, 4)   # [Batch, PredLen, TimeFeatures]

# 5. 推理
with torch.no_grad():
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    # outputs.shape: [1, 96, 7]  (Batch=1, PredLen=96, N_vars=7)

print(f"预测结果形状: {outputs.shape}")
print(f"预测值范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
```

---

#### 🔹 批量推理（测试集评估）

```python
from tqdm import tqdm
from utils.metrics import metric

# 1. 加载测试集
test_data, test_loader = data_provider(args, 'test')

# 2. 推理循环
preds = []
trues = []

model.eval()
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
        # 输入数据
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # Decoder 输入（前 label_len 取真值，后 pred_len 填零）
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # 推理
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 提取预测窗口
        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :]

        # 累积结果
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.cpu().numpy())

# 3. 拼接并计算指标
preds = np.concatenate(preds, axis=0)  # [N_samples, pred_len, N_vars]
trues = np.concatenate(trues, axis=0)

mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
```

---

### 4.3 评估指标详解

#### 📊 支持的评估指标（`utils/metrics.py`）

| 指标 | 公式 | 物理意义 | 代码位置 |
|------|------|---------|---------|
| **MAE** | $\frac{1}{N}\sum \|y_{pred} - y_{true}\|$ | 平均绝对误差（单位与原数据一致） | 第 14-15 行 |
| **MSE** | $\frac{1}{N}\sum (y_{pred} - y_{true})^2$ | 均方误差（对大误差敏感） | 第 18-19 行 |
| **RMSE** | $\sqrt{MSE}$ | 均方根误差（单位与原数据一致） | 第 22-23 行 |
| **MAPE** | $\frac{1}{N}\sum \|\frac{y_{pred} - y_{true}}{y_{true}}\|$ | 平均绝对百分比误差（0-1 范围） | 第 26-27 行 |
| **MSPE** | $\frac{1}{N}\sum (\frac{y_{pred} - y_{true}}{y_{true}})^2$ | 均方百分比误差 | 第 30-31 行 |

#### 📌 指标选择建议

- **长期预测（ETT/Weather/Traffic）：** 主要关注 **MSE** 和 **MAE**
- **M4 短期预测：** 使用 **SMAPE** 和 **MASE**（M4 竞赛标准）
- **对比不同模型：** 同时报告 MAE、MSE、RMSE 三项

#### 🔍 训练日志示例

```
Epoch: 1 cost time: 123.45
Epoch: 1 | Train Loss: 0.4523 Vali Loss: 0.3912 Test Loss: 0.4012
          MAE: 0.4567, MSE: 0.3912, RMSE: 0.6254, MAPE: 0.1234, MSPE: 0.0234

Epoch: 2 cost time: 119.32
Epoch: 2 | Train Loss: 0.3821 Vali Loss: 0.3654 Test Loss: 0.3701
          MAE: 0.4321, MSE: 0.3654, RMSE: 0.6045, MAPE: 0.1198, MSPE: 0.0221

...

EarlyStopping counter: 1 out of 10
Validation loss decreased (0.365400 --> 0.351200). Saving model ...
```

---

## 五、核心技术要点总结

### 5.1 Time-LLM 的三大创新

#### 🔹 1. Prompt-as-Prefix（提示作为前缀）

- **动态统计提示：** 提取 min/max/median/trend/lags 作为上下文
- **领域描述：** 从 `prompt_bank/` 加载数据集描述文本
- **联合嵌入：** Prompt Embeddings + Patch Embeddings 拼接后输入 LLM

#### 🔹 2. Reprogramming Layer（重编程层）

- **跨模态对齐：** 通过 Cross-Attention 将时序嵌入映射到 LLM 空间
- **参数高效：** 仅训练对齐层，LLM 完全冻结
- **知识迁移：** 利用 LLM 的语义理解能力增强时序建模

#### 🔹 3. Patching Strategy（分块策略）

- **局部性：** 保留时序局部依赖
- **效率：** 降低序列长度，减少计算复杂度
- **对齐：** Patch 类比文本 Token，符合 LLM 输入范式

---

### 5.2 与传统时序模型对比

| 维度 | 传统模型（Autoformer/DLinear） | Time-LLM |
|------|----------------------------|----------|
| **参数量** | 全量训练（100K-10M） | 仅训练对齐层（50-60M），LLM 冻结（117M-7B） |
| **数据需求** | 需要大量时序数据 | 可利用 LLM 预训练知识，小数据集也有效 |
| **泛化能力** | 依赖数据分布 | 跨数据集迁移能力更强 |
| **计算开销** | 训练快 | 推理时需加载 LLM（显存占用大） |
| **可解释性** | 基于注意力权重 | Prompt 提供统计解释 + Attention 可视化 |

---

### 5.3 低显存优化策略总结

| 优化项 | 原始配置 | 6GB 显存配置 | 优化效果 |
|--------|---------|-------------|---------|
| LLM 模型 | LLAMA-7B (14GB) | GPT-2 (500MB) | ★★★ 显存降低 96% |
| Batch Size | 24 | 2-4 | ★★★ 显存降低 83% |
| Seq Len | 512 | 96 | ★★ 显存降低 81% |
| LLM Layers | 32 | 6 | ★ 显存降低 81% |
| Mixed Precision | bf16 | fp16 | ★ 显存降低 50% |
| Num Workers | 10 | 2 | ★ CPU 内存优化 |

---

## 六、常见问题 (FAQ)

### Q1: 为什么 LLM 参数冻结后还能提升时序预测性能？

**A:** LLM 通过预训练学习了丰富的模式识别能力（例如序列依赖、长程关联）。虽然参数冻结，但通过 Reprogramming Layer 将时序数据"翻译"成 LLM 能理解的形式后，LLM 的表示能力依然可以被利用。类似于 Prompt Tuning，只调整输入而非模型权重。

---

### Q2: Mapping Layer 的作用是什么？

**A:** LLM 的词嵌入矩阵（如 GPT-2 的 50257 个词）太大，直接作为 Reprogramming 的 Source 会导致计算开销过高。Mapping Layer 将词表空间压缩为 1000 个可学习的"虚拟词"，作为 Reprogramming 的 Key/Value，既降低计算复杂度，又增强表达能力。

---

### Q3: 如何可视化 Reprogramming Layer 的对齐效果？

**A:** 可以提取 Attention 权重矩阵 `A` (shape: `[B*N, n_heads, num_patches, num_tokens]`)，绘制热力图：
```python
# 在 ReprogrammingLayer.reprogramming() 第 302 行后添加
self.attention_weights = A.detach().cpu()

# 训练后可视化
import matplotlib.pyplot as plt
plt.imshow(model.reprogramming_layer.attention_weights[0, 0, :, :], cmap='viridis')
plt.xlabel('Source Tokens (LLM Vocab)')
plt.ylabel('Time Series Patches')
plt.colorbar()
plt.show()
```

---

### Q4: 训练时显存 OOM 怎么办？

**A:** 按以下顺序调整：
1. **降低 Batch Size**（2 → 1）
2. **减少 LLM Layers**（6 → 4）
3. **缩短 Seq Len**（96 → 64）
4. **降低 d_ff**（32 → 16）
5. **启用梯度检查点**（在 `TimeLLM.py` 中添加 `self.llm_model.gradient_checkpointing_enable()`）

---

### Q5: 如何迁移到新数据集？

**A:**
1. 在 `dataset/prompt_bank/` 创建新的描述文本（如 `my_dataset.txt`）
2. 在 `data_provider/data_loader.py` 添加新的 Dataset 类
3. 在 `data_provider/data_factory.py` 注册新数据集
4. 在 `run_main.py` 使用 `--data my_dataset --prompt_domain 1` 启动训练

---

## 七、Qwen 2.5 3B 4-bit 量化支持 (2024-12-05 更新)

### 7.1 代码修改总结

#### 📁 `run_main.py` (第 82-84 行)
```python
# ========== 新增参数：支持本地模型路径和4-bit量化 ==========
parser.add_argument('--llm_model_path', type=str, default='', help='LLM model path (local or HuggingFace ID)')
parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization to save VRAM')
# =========================================================
```

#### 📁 `models/TimeLLM.py` (第 43-96 行)
```python
if configs.llm_model_path:
    # 通用模型加载逻辑
    from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
    
    quantization_config = None
    if configs.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    self.llm_model = AutoModel.from_pretrained(
        configs.llm_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto" if configs.load_in_4bit else None
    )
```

### 7.2 模型对比

| 模型 | 年份 | 参数量 | 16-bit 显存 | 4-bit 显存 | 性能 |
|------|------|--------|-------------|------------|------|
| GPT-2 | 2019 | 124M | ~500 MB | N/A | 基准 |
| **Qwen 2.5 3B** | 2024 | 3B | ~6 GB | **~1.5 GB** | **强得多** |
| Llama 3.1 8B | 2024 | 8B | ~16 GB | ~4.5 GB | 更强 |

### 7.3 4-bit 量化原理

**为什么 3B 模型能在 6GB 显存运行？**

| 精度 | 每参数字节 | 3B 模型显存 |
|------|-----------|-------------|
| FP32 | 4 字节 | 12 GB |
| FP16/BF16 | 2 字节 | 6 GB |
| **INT4 (NF4)** | **0.5 字节** | **1.5 GB** |

**NF4 量化配置：**
```python
BitsAndBytesConfig(
    load_in_4bit=True,              # 启用 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 FP16
    bnb_4bit_use_double_quant=True,  # 双重量化进一步压缩
    bnb_4bit_quant_type="nf4"        # 使用 NF4 (Normal Float 4-bit)
)
```

### 7.4 运行命令

```powershell
python run_main.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_512_96 ^
  --model_comment Qwen3B ^
  --model TimeLLM ^
  --data ETTm1 ^
  --features M ^
  --seq_len 512 ^
  --label_len 48 ^
  --pred_len 96 ^
  --batch_size 8 ^
  --d_model 32 ^
  --d_ff 32 ^
  --llm_dim 2048 ^
  --llm_model QWEN ^
  --llm_model_path "e:\timellm\Time-LLM\base_models\Qwen2.5-3B" ^
  --load_in_4bit
```

---

## 八、参考文献与资源

### 📚 论文原文
- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**
  ICLR 2024 | [arXiv:2310.01728](https://arxiv.org/abs/2310.01728)

### 🔗 相关链接
- **GitHub 仓库：** [https://github.com/KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM)
- **数据集下载：** [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
- **HuggingFace 模型：**
  - GPT-2: `openai-community/gpt2`
  - **Qwen 2.5 3B:** `Qwen/Qwen2.5-3B-Instruct` ★ 推荐
  - LLAMA-7B: `huggyllama/llama-7b`
  - BERT: `google-bert/bert-base-uncased`

### 🛠️ 推荐工具
- **显存监控：** `nvidia-smi -l 1` (Windows) / `watch -n 1 nvidia-smi` (Linux)
- **可视化：** `tensorboard --logdir=./logs`
- **调试：** `python -m pdb run_main.py ...`

---

**文档生成时间：** 2024-12-05
**最后更新：** 新增 Qwen 2.5 3B 4-bit 量化支持
**适用版本：** Time-LLM v1.0 (基于 ICLR'24 论文实现)
**作者：** Claude Code Technical Analysis
