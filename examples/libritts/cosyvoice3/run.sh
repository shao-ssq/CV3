#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# CosyVoice3 LibriTTS训练流程脚本

# 加载环境变量配置
. ./path.sh || exit 1;

# 流程控制参数：设置起始和结束阶段
stage=-1          # 起始阶段（-1表示从数据下载开始）
stop_stage=3      # 结束阶段

# 数据集配置
data_url=www.openslr.org/resources/60                              # LibriTTS数据集下载地址
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts              # 数据存储目录
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B  # 预训练模型路径

# ============================================================
# Stage -1: 数据下载
# 功能：从OpenSLR下载LibriTTS数据集的各个子集
# ============================================================
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  # 下载LibriTTS的7个子集：
  # - dev-clean/test-clean: 干净的开发/测试集
  # - dev-other/test-other: 其他条件的开发/测试集
  # - train-clean-100/360: 100/360小时的干净训练集
  # - train-other-500: 500小时的其他条件训练集
  for part in dev-clean test-clean train-clean-100; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

# ============================================================
# Stage 0: 数据准备
# 功能：生成训练所需的基础文件（wav.scp/text/utt2spk/spk2utt）
# 注意：CosyVoice3在序列中添加了指令（instruct）
# ============================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train-clean-100 dev-clean test-clean; do
    mkdir -p data/$x
    # NOTE in CosyVoice3, we add instruct in sequence
    # 为每个数据集添加指令提示词，用于引导模型生成
    python local/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir data/$x --instruct "You are a helpful assistant.<|endofprompt|>"
  done
fi

# ============================================================
# Stage 3: 转换为Parquet格式
# 功能：将准备好的数据转换为训练所需的parquet格式
# 注意：CosyVoice3支持在线特征提取，不再需要预先提取embedding和token
# ============================================================
# NOTE embedding/token extraction is not necessary now as we support online feature extraction
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train-clean-100 dev-clean test-clean; do
    mkdir -p data/$x/parquet
    # 每个parquet文件包含1000条语音，使用10个进程并行处理
    ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# ============================================================
# 训练配置参数
# ============================================================
# train llm
export CUDA_VISIBLE_DEVICES="0"                                    # 指定使用的GPU编号
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')  # 自动计算GPU数量
job_id=1986                                                        # 分布式训练任务ID
dist_backend="nccl"                                                # 分布式后端（NCCL适用于GPU）
num_workers=2                                                      # 数据加载的工作进程数
prefetch=100                                                       # 预取数据的批次数
train_engine=torch_ddp                                             # 训练引擎（torch_ddp或deepspeed）
# ============================================================
# Stage 5: 模型训练
# 功能：训练CosyVoice3的三个核心模块（llm, flow, hifigan）
# 支持：torch_ddp和deepspeed两种训练引擎
# ============================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # 合并训练集和验证集的数据列表
  cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
  cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list
  # 依次训练三个模块：llm（语言模型）、flow（流模型）、hifigan（声码器）
  for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice3/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice3/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# ============================================================
# Stage 6: 模型平均
# 功能：对多个checkpoint进行平均，提高模型稳定性和泛化能力
# 策略：选择验证集上表现最好的5个checkpoint进行平均
# ============================================================
# average model
average_num=5  # 平均最好的5个checkpoint
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best  # 选择验证集上最好的模型
  done
fi

# ============================================================
# Stage 7: 模型导出
# 功能：将训练好的模型导出为推理优化格式（JIT和ONNX）
# 用途：加速推理速度，便于部署
# 注意：需要先将训练好的llm或flow模型复制到model_dir
# ============================================================
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir   # 导出为TorchScript JIT格式
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir  # 导出为ONNX格式
fi