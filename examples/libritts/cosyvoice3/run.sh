#!/bin/bash
stage=-1
stop_stage=-1

# 设置 PYTHONPATH
export PYTHONPATH=/root/PycharmProjects/CV3:/root/PycharmProjects/CV3/third_party/Matcha-TTS:${PYTHONPATH:-}

# 解析命令行参数
if [ $# -ge 1 ]; then
    stage=$1
fi
if [ $# -ge 2 ]; then
    stop_stage=$2
fi

data_dir=/root/PycharmProjects/CV3/examples/aob/data
pretrained_model_dir=/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B

# ============================================================
# Stage 1: 数据准备
# 功能：生成训练所需的基础文件（wav.scp/text/utt2spk/spk2utt）
# 注意：CosyVoice3在序列中添加了指令（instruct）
# ============================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train dev test; do
    mkdir -p data/$x
    # 为每个数据集添加指令提示词，用于引导模型生成
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x --instruct "You are a helpful assistant.<|endofprompt|>"
  done
fi

# ============================================================
# Stage 2: 转换为Parquet格式
# 功能：将准备好的数据转换为训练所需的parquet格式
# 注意：CosyVoice3支持在线特征提取，不再需要预先提取embedding和token
# ============================================================
# NOTE embedding/token extraction is not necessary now as we support online feature extraction
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in {train,dev}; do
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
# Stage 3: 模型训练
# 功能：训练CosyVoice3的三个核心模块（llm, flow, hifigan）
# 支持：torch_ddp和deepspeed两种训练引擎
# 手动修改 examples/libritts/cosyvoice3/conf/cosyvoice3.yaml 下的 qwen_pretrain_path 为实际路径
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  pkill -9 -f "torchrun"
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # 合并训练集和验证集的数据列表
  cat data/train/parquet/data.list > data/train.data.list
  cat data/dev/parquet/data.list > data/dev.data.list

  # 检查数据文件是否为空
  if [ ! -s data/train.data.list ]; then
    echo "错误：训练数据列表为空 (data/train.data.list)"
    exit 1
  fi
  if [ ! -s data/dev.data.list ]; then
    echo "错误：验证数据列表为空 (data/dev.data.list)"
    echo "请先运行 stage 1 和 2 准备数据"
    exit 1
  fi

  # 依次训练三个模块：llm（语言模型）、flow（流模型）、hifigan（声码器）
  for model in llm flow hifigan; do
    echo "=========================================="
    echo "开始训练模型: $model"
    echo "=========================================="
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
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

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
      echo "错误：模型 $model 训练失败"
      exit 1
    fi
    echo "模型 $model 训练完成"
  done
fi

# ============================================================
# Stage 4: 模型平均
# 功能：对多个checkpoint进行平均，提高模型稳定性和泛化能力
# 策略：选择验证集上表现最好的5个checkpoint进行平均
# ============================================================
# average model
average_num=5  # 平均最好的5个checkpoint
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
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