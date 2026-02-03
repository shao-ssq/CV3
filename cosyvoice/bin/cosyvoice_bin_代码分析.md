1. `average_model.py` - 模型平均工具
2. `train.py` - 主训练脚本
3. `export_jit.py` - TorchScript导出工具
4. `export_onnx.py` - ONNX导出工具

### 1. average_model.py - 模型平均工具

**文件路径：** `cosyvoice/bin/average_model.py`

#### 作用
将多个训练检查点的模型参数进行加权平均，生成最优模型。这是一种常用的模型集成技术，可以减少过拟合，提高模型泛化能力。

#### 主要功能
- 扫描指定目录下的所有模型检查点文件（`*.pt`）
- 读取每个检查点对应的YAML配置文件，获取验证集loss
- 根据loss值排序，选出表现最好的N个模型
- 对选中模型的所有参数进行算术平均
- 保存平均后的模型权重到指定路径

#### 命令行参数
```bash
--dst_model      # 必需，输出的平均模型保存路径
--src_path       # 必需，源模型检查点所在目录
--val_best       # 可选，根据验证集loss选择最佳模型
--num            # 可选，平均模型的数量（默认5个）
```

#### 使用示例
```bash
python average_model.py \
  --src_path exp/llm/checkpoints \
  --dst_model exp/llm/avg_model.pt \
  --val_best \
  --num 5
```

#### 技术细节
- 排序规则：按loss从小到大排序（`reverse=False`）
- 参数平均：使用 `torch.true_divide` 进行精确除法
- 跳过字段：不对 `step` 和 `epoch` 进行平均

---

### 2. train.py - 主训练脚本 ⭐

**文件路径：** `cosyvoice/bin/train.py`

#### 作用
CosyVoice模型的完整训练流程实现，支持大规模分布式训练和多种训练策略。

#### 核心功能

##### 1) 分布式训练支持
- **PyTorch DDP**（Distributed Data Parallel）
- **DeepSpeed**：支持ZeRO优化，节省显存

##### 2) 多模型训练
支持训练不同组件：
- `llm`：大语言模型组件
- `flow`：Flow-based生成模型
- `hift`/`hifigan`：声码器（Vocoder）

##### 3) DPO训练（直接偏好优化）
- 支持 Direct Preference Optimization
- 需要提供参考模型（`--ref_model`）
- 使用 `DPOLoss` 进行优化

##### 4) 混合精度训练
- 使用 `--use_amp` 开启自动混合精度（AMP）
- 通过 `torch.cuda.amp.GradScaler` 进行梯度缩放

##### 5) 断点续训
- 支持从检查点恢复训练
- 自动加载 `step` 和 `epoch` 信息

##### 6) TensorBoard监控
- 实时记录训练损失
- 可视化训练指标

#### 命令行参数
```bash
# 必需参数
--model              # 训练的模型类型（llm/flow/hifigan等）
--config             # 配置文件路径
--train_data         # 训练数据文件
--cv_data            # 验证数据文件
--model_dir          # 模型保存目录

# 可选参数
--train_engine       # 训练引擎（torch_ddp/deepspeed，默认torch_ddp）
--ref_model          # DPO训练的参考模型
--checkpoint         # 断点续训的检查点路径
--qwen_pretrain_path # Qwen预训练模型路径
--tensorboard_dir    # TensorBoard日志目录
--num_workers        # 数据加载子进程数（默认0）
--prefetch           # 预取数据数量（默认100）
--use_amp            # 启用自动混合精度训练
--dpo                # 启用DPO训练
--timeout            # 分布式训练超时时间（默认60秒）
```

#### 训练流程
```
1. 解析命令行参数
2. 加载配置文件（使用hyperpyyaml）
3. 初始化分布式训练环境
4. 准备数据集和数据加载器
5. 初始化模型、优化器、学习率调度器
6. 加载检查点（如果提供）
7. 将模型分发到GPU
8. 保存初始检查点
9. 开始训练循环：
   - 每个epoch遍历训练数据
   - 定期在验证集上评估
   - 保存检查点
   - 记录TensorBoard日志
```

#### 使用示例
```bash
# 单GPU训练
python train.py \
  --model llm \
  --config conf/cosyvoice.yaml \
  --train_data data/train.json \
  --cv_data data/dev.json \
  --model_dir exp/llm

# 多GPU DDP训练
torchrun --nproc_per_node=4 train.py \
  --model llm \
  --config conf/cosyvoice.yaml \
  --train_data data/train.json \
  --cv_data data/dev.json \
  --model_dir exp/llm

# DeepSpeed训练
deepspeed train.py \
  --train_engine deepspeed \
  --deepspeed_config ds_config.json \
  --model llm \
  --config conf/cosyvoice.yaml \
  --train_data data/train.json \
  --cv_data data/dev.json \
  --model_dir exp/llm
```

#### 技术亮点
- 使用 `@record` 装饰器捕获分布式训练异常
- GAN训练有特殊初始化逻辑（判别器优化器）
- 支持Gloo和NCCL两种分布式后端
- 动态创建进程组并在每个epoch后销毁，避免超时问题

---

### 3. export_jit.py - TorchScript导出工具

**文件路径：** `cosyvoice/bin/export_jit.py`

#### 作用
将训练好的模型导出为 **TorchScript** 格式，用于生产环境高效推理部署。

#### 导出的模型组件

##### CosyVoice 模型导出内容：
1. **LLM文本编码器**
   - `llm.text_encoder.fp32.zip`（32位浮点）
   - `llm.text_encoder.fp16.zip`（16位浮点）

2. **LLM核心模块**
   - `llm.llm.fp32.zip`（保留 `forward_chunk` 方法）
   - `llm.llm.fp16.zip`（保留 `forward_chunk` 方法）

3. **Flow编码器**
   - `flow.encoder.fp32.zip`
   - `flow.encoder.fp16.zip`

##### CosyVoice2 模型导出内容：
- **Flow编码器**
  - `flow.encoder.fp32.zip`
  - `flow.encoder.fp16.zip`

#### 优化技术
```python
# 优化流程
torch.jit.script(model)               # 图编译
→ torch.jit.freeze(script)            # 冻结参数
→ torch.jit.optimize_for_inference()  # 推理优化
```

#### 命令行参数
```bash
--model_dir    # 模型目录路径（默认：pretrained_models/CosyVoice-300M）
```

#### 使用示例
```bash
python export_jit.py \
  --model_dir pretrained_models/CosyVoice-300M
```

#### TorchScript优势
- **性能提升**：编译后的图执行更快
- **部署友好**：不依赖Python运行时
- **跨平台**：可在C++环境中加载
- **移动端支持**：可部署到iOS/Android

#### 技术细节
- 设置融合策略：`torch._C._jit_set_fusion_strategy([('STATIC', 1)])`
- 关闭性能分析：避免额外开销
- 同时导出FP32和FP16：平衡精度和速度

---

### 4. export_onnx.py - ONNX导出工具

**文件路径：** `cosyvoice/bin/export_onnx.py`

#### 作用
将模型的 **Flow Decoder Estimator** 部分导出为 ONNX 格式，用于跨平台推理。

#### 主要功能

##### 1) 模型导出
- 导出组件：`flow.decoder.estimator`
- 输出文件：`flow.decoder.estimator.fp32.onnx`
- ONNX Opset版本：18

##### 2) 动态轴支持
支持可变序列长度：
```python
dynamic_axes={
    'x': {2: 'seq_len'},
    'mask': {2: 'seq_len'},
    'mu': {2: 'seq_len'},
    'cond': {2: 'seq_len'},
    'estimator_out': {2: 'seq_len'},
}
```

##### 3) 导出验证
- 使用 ONNX Runtime 加载导出的模型
- 生成10组随机输入进行测试
- 对比PyTorch和ONNX的输出，确保误差在容忍范围内
  - 相对误差：`rtol=1e-2`
  - 绝对误差：`atol=1e-4`

#### 输入张量
```python
x:    (batch_size, out_channels, seq_len)  # 噪声输入
mask: (batch_size, 1, seq_len)             # 掩码
mu:   (batch_size, out_channels, seq_len)  # 均值
t:    (batch_size)                         # 时间步
spks: (batch_size, out_channels)           # 说话人嵌入
cond: (batch_size, out_channels, seq_len)  # 条件特征
```

#### 命令行参数
```bash
--model_dir    # 模型目录路径（默认：pretrained_models/CosyVoice-300M）
```

#### 使用示例
```bash
python export_onnx.py \
  --model_dir pretrained_models/CosyVoice-300M
```

#### ONNX优势
- **跨平台部署**：Windows、Linux、macOS、移动端
- **多推理引擎**：ONNX Runtime、TensorRT、OpenVINO
- **硬件加速**：支持CPU、GPU、NPU等
- **语言无关**：可在C++、C#、Java等环境运行

#### 技术细节
- 使用 `@torch.no_grad()` 禁用梯度计算
- 设置 `do_constant_folding=True` 进行常量折叠优化
- 支持CUDA和CPU推理
- 使用tqdm显示验证进度

---

## 完整工作流程
```
┌─────────────────┐
│  1. 训练模型     │
│  (train.py)     │
└────────┬────────┘
         │
         ├─ 训练多个epoch
         ├─ 保存多个检查点
         │
┌────────▼────────┐
│  2. 模型平均     │
│  (average_model)│
└────────┬────────┘
         │
         ├─ 选择最佳5个模型
         ├─ 参数平均
         │
┌────────▼────────┐
│  3. 导出部署     │
└─────────────────┘
         │
         ├─ JIT导出 (export_jit.py)
         │  └─ 用于Python/C++部署
         │
         └─ ONNX导出 (export_onnx.py)
            └─ 用于跨平台/移动端部署
```