## 第一步：准备原始数据

### 1.1 准备音频文件和文本

**单说话人场景示例：**

假设你有以下原始数据：

```
raw_data/
├── audio_001.wav
├── audio_002.wav
├── audio_003.wav
├── audio_004.wav
└── transcripts.txt
```

### 1.2 创建数据目录

```bash
mkdir -p data/train
mkdir -p data/dev
```

### 1.3 创建基础文件

需要手动或通过脚本创建三个文件：

#### a) `wav.scp` - 音频文件路径映射

**格式：** `<utterance_id> <audio_file_path>`

```bash
# data/train/wav.scp（单说话人示例）
utt001 /absolute/path/to/raw_data/audio_001.wav
utt002 /absolute/path/to/raw_data/audio_002.wav
utt003 /absolute/path/to/raw_data/audio_003.wav
utt004 /absolute/path/to/raw_data/audio_004.wav
utt005 /absolute/path/to/raw_data/audio_005.wav
```

**注意：**
- utterance_id 必须唯一
- 必须使用**绝对路径**
- 音频格式：wav（推荐16kHz或24kHz采样率）
- 单说话人时，ID命名可以更简洁（utt001, utt002...）

#### b) `text` - 文本内容

**格式：** `<utterance_id> <text_content>`

```bash
# data/train/text
utt001 你好世界，这是第一句话
utt002 语音合成技术非常有趣
utt003 我们正在准备训练数据
utt004 单说话人模型训练更简单
utt005 希望训练效果很好
```

**注意：**
- 文本应与音频内容一致
- 可以包含中文、英文、标点符号
- utterance_id 必须与 wav.scp 中一致

#### c) `utt2spk` - utterance到说话人的映射

**格式：** `<utterance_id> <speaker_id>`

**单说话人场景（所有utterance映射到同一个说话人）：**

```bash
# data/train/utt2spk（单说话人示例）
utt001 my_speaker
utt002 my_speaker
utt003 my_speaker
utt004 my_speaker
utt005 my_speaker
```

**注意：**
- **单说话人时，所有行的speaker_id都相同**
- speaker_id 可以任意命名（如：my_speaker, target_voice, main_spk等）
- 这个文件用于提取说话人级别的平均嵌入

**运行：**
```bash
python create_basic_files.py
```
## 第二步：提取说话人嵌入

### 2.1 工具说明

**脚本位置：** `tools/extract_embedding.py`

**作用：** 从音频中提取说话人嵌入向量（Speaker Embedding），用于说话人识别和克隆。

**输入：**
- `wav.scp` - 音频文件路径
- `utt2spk` - utterance到说话人映射
- ONNX模型 - 说话人嵌入提取模型

**输出：**
- `utt2embedding.pt` - 每个utterance的嵌入向量
- `spk2embedding.pt` - 每个说话人的平均嵌入向量

### 2.2 下载嵌入模型

首先需要获取说话人嵌入提取模型（ONNX格式）。

**方法1：从预训练模型提取**
```bash
# 如果已有CosyVoice预训练模型
ls pretrained_models/CosyVoice-300M/campplus.onnx
```

**方法2：下载CamPlusPlus模型**
```bash
# CamPlusPlus是常用的说话人嵌入模型
# 可以从ModelScope或Hugging Face下载
wget https://modelscope.cn/models/xxx/campplus.onnx -O models/campplus.onnx
```

### 2.3 运行提取脚本

```bash
python tools/extract_embedding.py \
    --dir data/train \
    --onnx_path pretrained_models/CosyVoice-300M/campplus.onnx \
    --num_thread 8
```

**参数说明：**
- `--dir`：数据目录（包含wav.scp和utt2spk）
- `--onnx_path`：嵌入模型的ONNX文件路径
- `--num_thread`：并行线程数（建议4-16）

### 2.4 执行过程

脚本会：
1. 读取每个音频文件
2. 重采样到16kHz（如果需要）
3. 提取80维Fbank特征
4. 使用ONNX模型提取嵌入向量
5. 计算每个说话人的平均嵌入（多个utterance的平均值）
6. 保存结果到`.pt`文件

**进度显示：**
```
100%|████████████████████| 1000/1000 [02:15<00:00,  7.38it/s]
```

### 2.5 验证输出

```bash
ls -lh data/train/*.pt
```

应该看到：
```
-rw-r--r-- 1 user user 245K  utt2embedding.pt
-rw-r--r-- 1 user user  3.2K  spk2embedding.pt  # 单说话人时文件很小
```

**验证内容（单说话人版本）：**
```python
import torch

# 加载嵌入
utt2emb = torch.load("data/train/utt2embedding.pt")
spk2emb = torch.load("data/train/spk2embedding.pt")

print(f"Utterance数量: {len(utt2emb)}")
print(f"说话人数量: {len(spk2emb)}")
print(f"嵌入维度: {len(list(utt2emb.values())[0])}")

# ⭐ 单说话人特有检查
print(f"\n单说话人验证:")
print(f"说话人ID: {list(spk2emb.keys())}")
assert len(spk2emb) == 1, "单说话人场景应该只有1个说话人嵌入！"
print("✓ 单说话人验证通过")

# 输出示例（单说话人）：
# Utterance数量: 1000
# 说话人数量: 1  ← 单说话人
# 嵌入维度: 192
#
# 单说话人验证:
# 说话人ID: ['my_speaker']
# ✓ 单说话人验证通过
```

---

## 第三步：提取语音Token

### 3.1 工具说明

**脚本位置：** `tools/extract_speech_token.py`

**作用：** 从音频中提取语音token序列（用于某些高级训练模式）。

**输入：**
- `wav.scp` - 音频文件路径
- ONNX模型 - 语音token提取模型

**输出：**
- `utt2speech_token.pt` - 每个utterance的token序列

### 3.2 是否需要提取语音Token？

**需要提取的情况：**
- 训练CosyVoice2模型（使用LLM生成语音token）
- 使用DPO（Direct Preference Optimization）训练
- 需要离线提取token以加速训练

**不需要提取的情况：**
- 训练Flow模型（会在训练时动态计算）
- 训练HiFiGAN声码器

### 3.3 下载Token提取模型

```bash
# 通常在预训练模型目录中
ls pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx
```

### 3.4 运行提取脚本

```bash
python tools/extract_speech_token.py \
    --dir data/train \
    --onnx_path pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx \
    --num_thread 8
```

**参数说明：**
- `--dir`：数据目录（包含wav.scp）
- `--onnx_path`：token提取模型的ONNX文件路径
- `--num_thread`：并行线程数（建议4-16）

**注意：**
- 仅支持30秒以内的音频（长音频会被跳过）
- 需要GPU支持（使用CUDAExecutionProvider）

### 3.5 验证输出

```bash
ls -lh data/train/utt2speech_token.pt
```

```python
import torch

tokens = torch.load("data/train/utt2speech_token.pt")
print(f"Utterance数量: {len(tokens)}")

# 查看某个样本的token
utt_id = list(tokens.keys())[0]
print(f"样本 {utt_id} 的token数量: {len(tokens[utt_id])}")
print(f"Token序列示例: {tokens[utt_id][:10]}")
```

---

## 第四步：打包成Parquet格式

### 4.1 工具说明

**脚本位置：** `tools/make_parquet_list.py`

**作用：** 将所有数据整合成Parquet格式，便于高效训练。

**输入文件（在--src_dir目录下）：**
```
data/train/
├── wav.scp              ✓ 必需
├── text                 ✓ 必需
├── utt2spk              ✓ 必需
├── utt2embedding.pt     ✓ 必需
├── spk2embedding.pt     ✓ 必需
└── utt2speech_token.pt  ○ 可选
```

**输出文件（在--des_dir目录下）：**
```
data/train_parquet/
├── parquet_000000000.tar
├── parquet_000000001.tar
├── parquet_000000002.tar
├── ...
├── data.list            ← 这个文件用于训练！
├── utt2data.list
└── spk2data.list
```

### 4.2 运行打包脚本

```bash
python tools/make_parquet_list.py \
    --src_dir data/train \
    --des_dir data/train_parquet \
    --num_utts_per_parquet 1000 \
    --num_processes 8
```

**参数说明：**
- `--src_dir`：源数据目录（包含所有输入文件）
- `--des_dir`：输出目录（保存parquet文件）
- `--num_utts_per_parquet`：每个parquet包含的样本数（建议500-2000）
- `--num_processes`：并行进程数（建议4-16）

**可选参数：**
- `--dpo`：如果是DPO训练数据
- `--instruct`：如果包含指令文本

### 4.3 执行过程

```
100%|████████████████████| 10/10 [00:45<00:00,  4.52s/it]
spend time 45.2
```

脚本会：
1. 读取所有输入文件
2. 将音频文件转换为二进制数据
3. 按 `num_utts_per_parquet` 分片
4. 并行生成多个parquet文件
5. 创建 `data.list` 索引文件

### 4.4 验证输出

```bash
# 查看生成的文件
ls -lh data/train_parquet/

# 查看data.list内容
cat data/train_parquet/data.list
```

输出示例：
```
/absolute/path/to/data/train_parquet/parquet_000000000.tar
/absolute/path/to/data/train_parquet/parquet_000000001.tar
/absolute/path/to/data/train_parquet/parquet_000000002.tar
```

**验证parquet内容：**
```python
import pandas as pd

# 读取第一个parquet文件
df = pd.read_parquet("data/train_parquet/parquet_000000000.tar")

print(f"样本数量: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n第一个样本:")
print(df.iloc[0])

# 输出示例：
# 样本数量: 1000
# 列名: ['utt', 'wav', 'audio_data', 'text', 'spk', 'utt_embedding', 'spk_embedding', 'speech_token']
```

---

## 第五步：开始训练

### 5.1 准备训练集和验证集

重复上述步骤，分别准备训练集和验证集：

```bash
# 训练集
data/train_parquet/data.list  → 用于 --train_data

# 验证集
data/dev_parquet/data.list    → 用于 --cv_data
```

**建议验证集大小：** 训练集的1-5%

### 5.2 启动训练

```bash
# 单机4卡训练
torchrun --nproc_per_node=4 \
    cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --model flow \
    --config conf/cosyvoice.yaml \
    --train_data data/train_parquet/data.list \
    --cv_data data/dev_parquet/data.list \
    --model_dir exp/flow_model \
    --num_workers 4 \
    --prefetch 100 \
    --use_amp
```

训练将自动开始！

---
