# Time-LLM 训练命令详解 (Command Parameters Guide)

> **适用版本**: Qwen 2.5 3B + 4-bit 量化
> **硬件要求**: 6GB 显存
> **数据集**: ETTm1

---

## 📋 完整训练命令

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
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --batch_size 8 ^
  --d_model 32 ^
  --d_ff 32 ^
  --llm_dim 2048 ^
  --llm_layers 6 ^
  --num_workers 2 ^
  --prompt_domain 1 ^
  --train_epochs 10 ^
  --itr 1 ^
  --dropout 0.1 ^
  --llm_model QWEN ^
  --llm_model_path "e:\timellm\Time-LLM\base_models\Qwen2.5-3B" ^
  --load_in_4bit
```

---

## 📖 参数详解

### 1️⃣ 基础配置 (Basic Config)

#### `--task_name long_term_forecast`
- **含义**: 任务类型
- **代码位置**: `run_main.py` 第 30 行
- **可选值**:
  - `long_term_forecast` (长期预测，推荐)
  - `short_term_forecast` (短期预测)
  - `imputation` (缺失值填充)
  - `classification` (分类)
  - `anomaly_detection` (异常检测)
- **作用**: 决定模型的输出格式和损失函数类型

#### `--is_training 1`
- **含义**: 训练模式开关
- **代码位置**: `run_main.py` 第 32 行
- **可选值**: `1` (训练), `0` (测试)
- **作用**: 控制是训练还是仅推理评估

#### `--model_id ETTm1_512_96`
- **含义**: 模型标识符
- **代码位置**: `run_main.py` 第 33 行
- **格式**: `{数据集}_{输入长度}_{预测长度}`
- **作用**: 生成唯一的 checkpoint 文件夹名称

#### `--model_comment Qwen3B`
- **含义**: 模型备注
- **代码位置**: `run_main.py` 第 34 行
- **作用**: 附加到 checkpoint 名称末尾，便于区分不同实验

#### `--model TimeLLM`
- **含义**: 选择模型架构
- **代码位置**: `run_main.py` 第 35 行
- **可选值**: `TimeLLM`, `Autoformer`, `DLinear`
- **作用**: 决定使用哪个模型类
- **对应代码**: `run_main.py` 第 138 行 → `model = TimeLLM.Model(args).float()`

---

### 2️⃣ 数据配置 (Data Config)

#### `--data ETTm1`
- **含义**: 数据集名称
- **代码位置**: `run_main.py` 第 40 行
- **可选值**: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `ECL`, `Traffic`
- **作用**: 通过 `data_provider/data_factory.py` 路由到对应数据集类
  - ETTm1/ETTm2 → `Dataset_ETT_minute`
  - ETTh1/ETTh2 → `Dataset_ETT_hour`

#### `--root_path ./dataset/ETT-small/`
- **含义**: 数据集根目录
- **代码位置**: `run_main.py` 第 41 行
- **作用**: 指定数据文件存放位置
- **完整路径**: `{root_path}/{data_path}` = `./dataset/ETT-small/ETTm1.csv`

#### `--data_path ETTm1.csv`
- **含义**: 数据文件名
- **代码位置**: `run_main.py` 第 42 行
- **作用**: 与 `root_path` 拼接成完整路径

#### `--features M`
- **含义**: 预测任务类型
- **代码位置**: `run_main.py` 第 43 行
- **可选值**:
  - `M` (Multivariate): 多变量预测多变量 (7→7)
  - `S` (Single): 单变量预测单变量 (1→1)
  - `MS` (Multivariate-to-Single): 多变量预测单变量 (7→1)
- **作用**: 决定输入输出的变量数量

---

### 3️⃣ 时序长度配置 (Sequence Length)

#### `--seq_len 512`
- **含义**: 输入序列长度（历史观测窗口）
- **代码位置**: `run_main.py` 第 56 行
- **作用**: 模型观察过去 512 个时间步
- **显存影响**: seq_len 越大，显存占用越高
- **推荐值**: 6GB 显存下推荐 96-512

#### `--label_len 48`
- **含义**: Decoder 起始 token 长度
- **代码位置**: `run_main.py` 第 57 行
- **作用**: Decoder 输入的前 48 步使用真值，后面填零
- **推荐值**: 通常设为 `pred_len / 2`

#### `--pred_len 96`
- **含义**: 预测长度（未来预测窗口）
- **代码位置**: `run_main.py` 第 58 行
- **作用**: 模型预测未来 96 个时间步
- **常用值**: 96, 192, 336, 720

**数据流示例**:
```
输入: [过去 512 步] → 输出: [未来 96 步]
```

---

### 4️⃣ 模型结构配置 (Model Architecture)

#### `--enc_in 7`
- **含义**: Encoder 输入变量数
- **代码位置**: `run_main.py` 第 62 行
- **作用**: ETT 数据集有 7 个特征 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)

#### `--dec_in 7`
- **含义**: Decoder 输入变量数
- **代码位置**: `run_main.py` 第 63 行
- **作用**: 与 `enc_in` 保持一致

#### `--c_out 7`
- **含义**: 输出变量数
- **代码位置**: `run_main.py` 第 64 行
- **作用**: features=M 时，c_out=7（预测全部 7 个变量）

#### `--d_model 32`
- **含义**: 模型隐藏层维度（Patch Embedding 维度）
- **代码位置**: `run_main.py` 第 65 行
- **作用**: 控制 PatchEmbedding 的输出维度
- **显存影响**: d_model 越大，可训练参数越多
- **推荐值**: 6GB 显存下推荐 16-32

#### `--n_heads 8`
- **含义**: 多头注意力的头数
- **代码位置**: `run_main.py` 第 66 行
- **作用**: Reprogramming Layer 的注意力头数
- **默认值**: 8

#### `--e_layers 2`
- **含义**: Encoder 层数
- **代码位置**: `run_main.py` 第 67 行
- **作用**: Time-LLM 中保留兼容性参数

#### `--d_layers 1`
- **含义**: Decoder 层数
- **代码位置**: `run_main.py` 第 68 行
- **作用**: Time-LLM 中保留兼容性参数

#### `--factor 3`
- **含义**: 注意力因子
- **代码位置**: `run_main.py` 第 71 行
- **作用**: Time-LLM 中保留兼容性参数

#### `--d_ff 32`
- **含义**: Feed-Forward 网络维度
- **代码位置**: `run_main.py` 第 69 行
- **作用**:
  - 控制 LLM 输出截取的维度数
  - `dec_out = dec_out[:, :, :d_ff]` (TimeLLM.py 第 301 行)
  - 影响 FlattenHead 输入维度: `head_nf = d_ff * num_patches`
- **推荐值**: 6GB 显存下推荐 32

#### `--dropout 0.1`
- **含义**: Dropout 概率
- **代码位置**: `run_main.py` 第 72 行
- **作用**: 防止过拟合

---

### 5️⃣ LLM 配置 (LLM Config) ⭐ 最关键

#### `--llm_model QWEN`
- **含义**: LLM 基座模型类型
- **代码位置**: `run_main.py` 第 80 行
- **可选值**: `LLAMA`, `GPT2`, `BERT`, `QWEN`
- **作用**: 当设置 `llm_model_path` 时，此参数仅用于标识
- **说明**: 使用 `llm_model_path` 时，实际加载由 `AutoModel` 完成

#### `--llm_dim 2048` ⭐
- **含义**: LLM 隐藏层维度
- **代码位置**: `run_main.py` 第 81 行
- **不同 LLM 的维度**:
  - **Qwen 2.5 3B: 2048** ✅
  - Llama-7B: 4096
  - GPT-2: 768
  - BERT: 768
- **作用**:
  - 决定 Reprogramming Layer 的输出维度
  - 决定 Prompt Embeddings 的维度
  - **必须与模型的 `config.hidden_size` 匹配！**
- **对应代码**: `models/TimeLLM.py` 第 39 行 `self.d_llm = configs.llm_dim`

#### `--llm_model_path "e:\timellm\Time-LLM\base_models\Qwen2.5-3B"` ⭐
- **含义**: LLM 模型本地路径或 HuggingFace ID
- **代码位置**: `run_main.py` 第 83 行（新增参数）
- **作用**:
  - 指定本地模型路径，使用通用加载逻辑 (`AutoModel`)
  - 支持任意 HuggingFace 模型
  - 若为空，则使用 `llm_model` 参数指定的硬编码路径
- **对应代码**: `models/TimeLLM.py` 第 43-96 行

#### `--load_in_4bit` ⭐⭐⭐
- **含义**: 启用 4-bit 量化
- **代码位置**: `run_main.py` 第 84 行（新增参数）
- **作用**:
  - 使用 BitsAndBytesConfig 进行 NF4 量化
  - **显存从 ~6GB 降至 ~1.5GB！**
  - 适配 6GB 显卡的核心配置
- **量化配置**（`models/TimeLLM.py` 第 52-57 行）:
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"
  )
  ```

#### `--llm_layers 6` ⭐⭐⭐ 最重要！
- **含义**: 使用的 LLM 层数
- **代码位置**: `run_main.py` 第 101 行
- **作用**:
  - 设置 LLM config 的 `num_hidden_layers`
  - 控制 LLM 的深度，直接影响显存占用
- **为什么必须指定？**
  - Qwen 2.5 3B 原始有 **32 层**
  - 不指定会使用全部 32 层，导致 **显存爆炸（~8GB）**
  - 设为 6 层可将显存降至 **~1.5GB**
- **对应代码**: `models/TimeLLM.py` 第 60 行
  ```python
  self.llm_config.num_hidden_layers = configs.llm_layers
  ```
- **推荐值**: 6GB 显存下推荐 **6**

---

### 6️⃣ 训练优化配置 (Training Config)

#### `--batch_size 8`
- **含义**: 训练批大小
- **代码位置**: `run_main.py` 第 92 行
- **作用**: 一次训练处理的样本数
- **显存影响**: batch_size 越大，显存占用越高
- **推荐值**:
  - Qwen 2.5 3B + 4-bit: **8-16**
  - 若 OOM，降至 **4 或 2**

#### `--num_workers 2` ⭐
- **含义**: 数据加载器的进程数
- **代码位置**: `run_main.py` 第 88 行
- **默认值**: 10（太高！）
- **作用**: 控制数据预处理的并行度
- **为什么要设为 2？**
  - 默认值 10 会启动 10 个数据加载进程
  - 占用过多 CPU 内存，可能导致系统卡顿
  - 6GB 显存环境推荐 **2**

#### `--train_epochs 10`
- **含义**: 训练轮数
- **代码位置**: `run_main.py` 第 90 行
- **作用**: 完整遍历训练集的次数
- **推荐值**: 10-100（取决于数据集大小）

#### `--itr 1`
- **含义**: 实验重复次数
- **代码位置**: `run_main.py` 第 89 行
- **作用**: 重复训练多次以计算统计量
- **对应代码**: `run_main.py` 第 109 行（外层循环 `for ii in range(args.itr)`）

#### `--prompt_domain 1` ⭐⭐⭐
- **含义**: 是否使用领域特定提示词
- **代码位置**: `run_main.py` 第 79 行
- **作用**:
  - `1`: 从 `dataset/prompt_bank/ETT.txt` 加载描述 ✅
  - `0`: 使用硬编码的默认描述
- **为什么必须设为 1？**
  - 不设置会使用不完整的默认描述
  - 无法加载你修复后的 ETT.txt（包含 ETTm2）
- **对应代码**: `models/TimeLLM.py` 第 223-226 行
  ```python
  if configs.prompt_domain:
      self.description = configs.content  # 从 ETT.txt 加载
  else:
      self.description = 'The Electricity Transformer Temperature...'  # 硬编码
  ```
- **加载流程**: `run_main.py` 第 142 行 → `utils/tools.py` 第 226-233 行

---

## 🔗 参数与代码对应关系

### 关键数据流路径

```
1️⃣ 参数解析
   run_main.py 第 104 行: args = parser.parse_args()

2️⃣ 加载领域提示词
   run_main.py 第 142 行: args.content = load_content(args)
   → utils/tools.py 第 226-233 行
   → 读取 dataset/prompt_bank/ETT.txt

3️⃣ 加载数据
   run_main.py 第 129-131 行:
   train_data, train_loader = data_provider(args, 'train')
   → data_provider/data_factory.py

4️⃣ 初始化模型
   run_main.py 第 138 行: model = TimeLLM.Model(args).float()
   → models/TimeLLM.py 第 30-250 行

5️⃣ 模型加载（Qwen 2.5 3B）
   models/TimeLLM.py 第 43-96 行:
   - 检测到 llm_model_path 不为空
   - 使用 AutoModel.from_pretrained()
   - 应用 4-bit 量化配置（第 50-57 行）
   - 设置层数（第 60 行）

6️⃣ 训练循环
   run_main.py 第 150-结束
```

---

## 📊 显存占用估算

| 组件 | 显存占用 | 参数控制 |
|------|---------|---------|
| Qwen 2.5 3B (4-bit, 6层) | ~1.5 GB | `--load_in_4bit`, `--llm_layers` |
| Mapping Layer | ~0.2 GB | `--llm_dim` |
| Reprogramming Layer | ~0.1 GB | `--d_model`, `--llm_dim` |
| Patch Embeddings | ~0.05 GB | `--d_model`, `--seq_len` |
| 中间变量 (batch=8) | ~2.0 GB | `--batch_size`, `--seq_len` |
| 梯度缓存 | ~0.5 GB | 可训练参数量 |
| 系统占用 | ~0.65 GB | - |
| **总计** | **~5.0 GB** | ✅ 6GB 显存足够！ |

---

## ⚠️ 新增参数说明

以下 **6 个参数** 是在你原始命令基础上新增的，均为**必需或强烈推荐**：

| 参数 | 重要性 | 原因 |
|------|--------|------|
| `--llm_layers 6` | ⭐⭐⭐ 必需 | 不设置会使用 32 层，显存爆炸 |
| `--prompt_domain 1` | ⭐⭐⭐ 必需 | 不设置无法加载 ETT.txt 描述 |
| `--num_workers 2` | ⭐⭐ 强烈推荐 | 默认 10 会占用过多 CPU 内存 |
| `--train_epochs 10` | ⭐ 推荐 | 明确指定训练轮数 |
| `--itr 1` | ⭐ 推荐 | 明确指定实验次数 |

---

## 🚨 常见问题与解决

### 问题 1：OOM (显存不足)
**症状**: `CUDA out of memory`

**解决方案**（按优先级）:
1. 降低 `--batch_size` 为 4 或 2
2. 降低 `--seq_len` 为 256
3. 降低 `--llm_layers` 为 4

### 问题 2：提示词未加载
**症状**: 训练时未使用 ETT.txt 描述

**解决方案**:
- 确保设置 `--prompt_domain 1`
- 检查 `dataset/prompt_bank/ETT.txt` 是否存在

### 问题 3：模型加载失败
**症状**: `Model files not found`

**解决方案**:
- 确认 `base_models/Qwen2.5-3B/` 下有完整的 `.safetensors` 文件
- 确认 `config.json` 和 `tokenizer.json` 存在

---

## 📝 生成的 Checkpoint 路径

根据上述参数，checkpoint 将保存到：

```
checkpoints/long_term_forecast_ETTm1_512_96_TimeLLM_ETTm1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-Qwen3B/checkpoint
```

**路径解析**:
- `long_term_forecast`: task_name
- `ETTm1_512_96`: model_id
- `TimeLLM`: model
- `ftM`: features=M
- `sl512`: seq_len=512
- `pl96`: pred_len=96
- `dm32`: d_model=32
- `test_0`: des=test, itr=0
- `Qwen3B`: model_comment

---

**文档生成时间**: 2024-12-08
**适配硬件**: 6GB VRAM GPU
**适配模型**: Qwen 2.5 3B + 4-bit NF4 量化
