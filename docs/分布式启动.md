# Token2Wav 分布式部署指南

## 1. 架构概述
### 分布式架构（DISTRIBUTED_MODE=True）
```
┌─────────────────────┐          ┌─────────────────────────────┐
│     机器 A (LLM)    │          │     机器 B (Token2Wav)      │
│                     │  gRPC    │                             │
│  ┌───────────────┐  │ ──────>  │  ┌─────────┐  ┌─────────┐  │
│  │  LLM Service  │  │  tokens  │  │ GPU:0   │  │ GPU:1   │  │
│  │  (vLLM)       │  │          │  │ Service │  │ Service │  │
│  │  端口: 50000  │  │          │  │ :50001  │  │ :50002  │  │
│  └───────────────┘  │          │  └─────────┘  └─────────┘  │
└─────────────────────┘          └─────────────────────────────┘
         ▲
         │ gRPC
    ┌────┴────┐
    │ Client  │
    └─────────┘
```

```bash
# 终端 1: Token2Wav Service (GPU:1)
CUDA_VISIBLE_DEVICES=1 python token2wav_server.py --port 50001 --load_trt --fp16

# 终端 2: Token2Wav Service (GPU:2)
CUDA_VISIBLE_DEVICES=2 python token2wav_server.py --port 50002 --load_trt --fp16

# 终端 3: LLM Service (GPU:0)
CUDA_VISIBLE_DEVICES=0 python server.py --port 50000 --fp16

# 测试
python client.py --port 50000 --mode zero_shot_by_spk_id --spk_id 001 --stream --tts_text "测试"
```
