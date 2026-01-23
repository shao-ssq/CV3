# vllm settings
# from vllm.engine.arg_utils import AsyncEngineArgs
# AsyncEngineArgs
ENGINE_ARGS = {
    # "enforce_eager": True,
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 2048,
    "max_num_seqs": 256,
    "disable_log_stats": True,
    "dtype": "float16",
}

from vllm.sampling_params import RequestOutputKind

# SamplingParams
SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_p": 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_k": 25,      # 与 CosyVoice 保持一致，优化采样质量
    # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
    # "presence_penalty": 1.0,    # 不支持设置
    # "frequency_penalty": 0.0,   # 不支持设置
    "detokenize": False,  # 目前 vllm 0.7.3 v1版本中设置无效，待后续版本更新后减少计算
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
}

# 设置frontend中 ZhNormalizer 的 overwrite_cache 参数
# 首次运行时，需要设置 True 正确生成缓存，避免 frontend 过滤掉儿化音。
# 后续可以设置为 False 可避免后续运行时重复生成。
OVERWRITE_NORMALIZER_CACHE = True

# 限制 estimator 内存方法  由 @hexisyztem 提供
# 原本代码编译后的flow trt模型 显存占用4.6G过大，修改为 1.6G，便于启动多个 estimator 实例，并发推理。
# 修改 cosyvoice/utils/file_utils.py:64
#     # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)  # 8GB
#     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
# 删除已经编译的模型./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp16.mygpu.plan
# 重新运行服务器代码  --trt 将重新编译模型，到时将生成新的模型，并单个 estimator 只占用 1.6GB 显存
# 根据GPU显存大小量及性能设置合适的 ESTIMATOR_COUNT
ESTIMATOR_COUNT = 4

# ============ 分布式部署配置 ============
# 是否启用分布式模式（LLM 和 Token2Wav 分离部署）
DISTRIBUTED_MODE = False

# Token2Wav 服务列表
TOKEN2WAV_SERVICES = [
    {"host": "localhost", "port": 50001},
    # {"host": "localhost", "port": 50002},
    # {"host": "192.168.1.100", "port": 50001},
]

# 负载均衡策略: round_robin | session_sticky
# 流式推理必须使用 session_sticky
LOAD_BALANCE_STRATEGY = "session_sticky"

# Token2Wav 服务超时 (毫秒)
TOKEN2WAV_TIMEOUT_MS = 30000

# ============ WebSocket 和 HTTP 服务配置 ============
import logging
import os

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TRT并发数
trt_concurrent = 4

# 录音上传目录
record_upload_dir = os.path.join(os.getcwd(), "uploads", "records")
os.makedirs(record_upload_dir, exist_ok=True)

# 录音哈希分桶数量
record_hash_count = 100

# 句子音频保存目录
sentence_audio_dir = os.path.join(os.getcwd(), "outputs", "sentences")
os.makedirs(sentence_audio_dir, exist_ok=True)

# 是否保存句子音频
save_sentence_audio = False

# 是否缓存句子
cache_sentence = True