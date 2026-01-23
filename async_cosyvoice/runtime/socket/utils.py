import base64
import hashlib
import json
import os
import shutil
import struct
from io import BytesIO
from typing import AsyncGenerator, Any, AsyncIterator

import lameenc
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from hyperpyyaml import load_hyperpyyaml
from starlette.concurrency import run_in_threadpool

from async_cosyvoice.config import logger, auth_info


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    # if sample_rate < target_sr:
    #     raise SpeakerAddException(f"发音人音频采样率不能低于{target_sr}")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def load_wav_and_upsample(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def load_audio_from_bytes(audio_data, target_sr):
    # 将字节数据包装成文件对象
    buffer = BytesIO(audio_data)
    # 使用soundfile后端加载音频
    speech, sample_rate = torchaudio.load(buffer, backend='soundfile')
    # 多声道转单声道（取均值）
    speech = speech.mean(dim=0, keepdim=True)

    # 检查并调整采样率
    if sample_rate != target_sr:
        if sample_rate < target_sr:
            raise ValueError(f"原始采样率 {sample_rate}Hz 必须不低于目标采样率 {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        speech = resampler(speech)
    return speech


class AsyncWrapper:
    def __init__(self, obj):
        self.obj = obj

    async def __aiter__(self):
        for item in self.obj:
            yield item


def _tensor_to_bytes(tensor: torch.Tensor, format: str = None, sample_rate=24000):
    # 统一Tensor形状为 (channels, samples)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        if tensor.size(0) > tensor.size(1):  # 假设输入为 (samples, channels)
            tensor = tensor.permute(1, 0)
    else:
        raise ValueError("Invalid tensor shape")

    match format:
        case "mp3":
            return _encode_mp3(tensor, sample_rate)
        case "wav":
            # bits_per_sample: PCM/WAV的量化位数（16或32）
            return _encode_wav(tensor, sample_rate, 16)
        case "pcm":
            return _encode_pcm(tensor, 16)
        case None:
            return tensor.numpy().astype(np.float32).tobytes()
        case _:
            raise ValueError(f"Unsupported format: {format}")


def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.float32))).unsqueeze(0)


async def convert_audio_tensor_to_bytes(
        tensor_generator: torch.Tensor | AsyncGenerator[torch.Tensor, Any],
        format: str = None, sample_rate=24000,
        stream=False) -> AsyncGenerator[bytes, Any]:
    """将音频Tensor转换为指定格式的字节流

    Args:
        tensor_generator: 输入音频Tensor，形状需为 (channels, samples) 或 (samples,)
        format: 目标格式，支持 'wav', 'pcm', 'mp3'
        sample_rate: 采样率（默认16000）
        stream: 是否以流式方式返回音频数据

    Returns:
        bytes: 编码后的音频字节流
        AsyncGenerator[bytes]: 流式返回音频数据
    """
    if isinstance(tensor_generator, torch.Tensor):
        tensor_generator = AsyncWrapper([tensor_generator])
    if not stream:
        tensor: torch.Tensor | None = None
        async for chunk in tensor_generator:
            if tensor is not None:
                tensor = torch.concat([tensor, chunk], dim=1)
            else:
                tensor = chunk

        yield await run_in_threadpool(_tensor_to_bytes, tensor, format, sample_rate)
    else:
        channels, bit_depth = 1, 16
        bitrate = 128
        if format == 'mp3':
            async for data in _encode_mp3_stream(tensor_generator, sample_rate, channels, bitrate):
                yield data
        elif format == 'pcm':
            # 无头，直接输出原始PCM
            async for chunk in tensor_generator:
                yield _pcm_to_bytes(chunk, bit_depth)
        elif format == 'wav':
            # 生成WAV头
            yield _generate_wav_header(sample_rate, channels, bit_depth)
            # 直接输出PCM数据
            async for chunk in tensor_generator:
                yield _pcm_to_bytes(chunk, bit_depth)
        else:
            raise ValueError(f"不支持的格式：{format}")


def _encode_wav(tensor: torch.Tensor, sr: int, bits: int) -> bytes:
    """编码WAV格式"""
    if bits == 16:
        encoding = "PCM_S"
        bits_depth = 16
    elif bits == 32:
        encoding = "PCM_F"
        bits_depth = 32
    else:
        raise ValueError("Only 16/32-bit WAV supported")
    buffer = BytesIO()
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="wav",
        encoding=encoding,
        bits_per_sample=bits_depth,
    )
    buffer.seek(0)
    return buffer.getvalue()


def _encode_pcm(tensor: torch.Tensor, bits: int) -> bytes:
    """编码原始PCM数据"""
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.float64, "输入张量应为浮点类型"
    np_array = tensor.cpu().numpy()
    np_array = np.clip(np_array, -1.0, 1.0)

    # 量化到目标位深
    if bits == 16:
        np_array = (np_array * 32767.0).astype(np.int16)
    elif bits == 32:
        np_array = (np_array * 2147483647.0).astype(np.int32)
    else:
        raise ValueError("Only 16/32-bit PCM supported")
    return np_array.tobytes()


def _encode_mp3(tensor: torch.Tensor, sr: int) -> bytes:
    """编码MP3格式"""
    buffer = BytesIO()

    # 注意：需要安装支持MP3编码的后端（如ffmpeg, libsox）
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="mp3",
        encoding="MP3",
    )
    buffer.seek(0)
    return buffer.getvalue()


def _generate_wav_header(sample_rate: int, channels: int = 1, bit_depth: int = 16) -> bytes:
    """生成适用于流式传输的WAV头，使用0xFFFFFFFF表示无限长度。

    Args:
        sample_rate (int): 采样率
        channels (int, optional): 通道数. Defaults to 1.
        bit_depth (int, optional): 位深（16或32位）。 Defaults to 16.
    """
    byte_rate = sample_rate * channels * (bit_depth // 8)
    block_align = channels * (bit_depth // 8)

    header = b'RIFF' + \
             struct.pack('<I', 0xFFFFFFFF) + \
             b'WAVE' + \
             b'fmt ' + \
             struct.pack('<IHHIIHH', 16, 1, channels, sample_rate, byte_rate, block_align, bit_depth) + \
             b'data' + \
             struct.pack('<I', 0xFFFFFFFF)  # 数据大小（未知）
    return header


def _pcm_to_bytes(pcm_data: torch.Tensor, bit_depth: int = 16) -> bytes:
    """将PCM tensor转换为字节"""
    if bit_depth not in (16, 32):
        raise ValueError(f"不支持的位深度：{bit_depth}")
    if bit_depth == 16:
        if pcm_data.dtype != torch.int16:
            pcm_data = pcm_data.to(torch.float32)
            pcm_data = (pcm_data * 32767.0).clamp(-32768, 32767).to(torch.int16)
        return pcm_data.numpy().tobytes()
    elif bit_depth == 32:
        if pcm_data.dtype != torch.float32:
            pcm_data = pcm_data.to(torch.float32) / 32768.0
        return pcm_data.numpy().tobytes()


async def _encode_mp3_stream(
        audio_chunks: AsyncIterator[torch.Tensor],
        sample_rate: int = 24000,
        channels: int = 1,
        bitrate: int = 128,
) -> AsyncGenerator[bytes, None]:
    """MP3编码实现"""
    encoder = lameenc.Encoder()
    encoder.set_channels(channels)
    encoder.set_bit_rate(bitrate * 1000)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_out_sample_rate(sample_rate)
    encoder.set_quality(9)
    encoder.set_vbr(4)
    encoder.set_vbr_quality(9)
    async for chunk in audio_chunks:
        pcm_bytes = _pcm_to_bytes(chunk, 16)  # MP3通常使用16位
        mp3_data = encoder.encode(pcm_bytes)
        if mp3_data:
            yield bytes(mp3_data)
    # 刷新编码器缓冲区
    final_data = encoder.flush()
    if final_data:
        yield bytes(final_data)


def generate_base64_data(model_output):
    audio_base64 = base64.b64encode(model_output).decode('utf-8')
    return audio_base64


async def generator_wrapper(audio_data_generator: AsyncGenerator[dict, None], sample_rate: int = 24000) -> \
        AsyncGenerator[torch.Tensor, None]:
    async for chunk in audio_data_generator:
        audio_tensor = chunk["tts_speech"]
        if sample_rate == 24000:
            yield audio_tensor
            continue

        re_sampler = T.Resample(24000, sample_rate, dtype=audio_tensor.dtype)
        yield re_sampler(audio_tensor)


def get_content_type(fmt: str, sample_rate: int) -> str:
    """获取对应格式的Content-Type"""
    if fmt == "pcm":
        return f"audio/L16; rate={sample_rate}; channels=1"
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "wav": "audio/wav"
    }[fmt]


def calculate_text_md5(text: str):
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode('utf-8'))
    return md5_hash.hexdigest()


def hash_index(key: str, num_buckets: int):
    # 将key转换为字符串并编码
    key_str = str(key).encode('utf-8')
    # 计算哈希值 (使用MD5、SHA1等都可以)
    hash_value = hashlib.md5(key_str).hexdigest()
    # 将哈希值转换为整数
    hash_int = int(hash_value, 16)
    # 分配到指定数量的桶中
    return hash_int % num_buckets


def export_cosyvoice2_vllm(model_dir, device):
    export_path = os.path.join(model_dir, "vllm")
    if os.path.exists(export_path):
        return
    pretrained_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    os.makedirs(export_path, exist_ok=True)

    hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
    if not os.path.exists(hyper_yaml_path):
        raise ValueError('{} not found!'.format(hyper_yaml_path))
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f,
                                   overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    dtype = torch.bfloat16
    model = configs["llm"]
    model.load_state_dict(torch.load(f"{model_dir}/llm.pt", map_location=device), strict=True)
    model.to(device).eval()
    model.llm.model.to(device)
    model.llm.model.to(dtype)

    logger.info(model.llm.model.config)
    pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
    vocab_size = model.llm.model.config.vocab_size
    speech_token_size = model.speech_embedding.num_embeddings
    feature_size = model.speech_embedding.embedding_dim
    llm_embedding_size = model.llm_embedding.num_embeddings
    pad_speech_size = ((speech_token_size + pad_to - 1) // pad_to) * pad_to
    pad_vocab_size = ((vocab_size + speech_token_size + 3 + pad_to - 1) // pad_to) * pad_to

    # lm_head
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=pad_speech_size, bias=True)
    logger.info(f"model.llm_decoder.weight shape {model.llm_decoder.weight.shape}")
    logger.info(f"model.llm_decoder.bias shape {model.llm_decoder.bias.shape}")
    with torch.no_grad():
        new_lm_head.weight[:speech_token_size] = model.llm_decoder.weight
        new_lm_head.bias[:speech_token_size] = model.llm_decoder.bias
        new_lm_head.weight[speech_token_size:] = 0
        new_lm_head.bias[speech_token_size:] = 0
    model.llm.model.lm_head = new_lm_head

    model.llm.model.text_embedding = model.llm.model.model.embed_tokens

    new_codec_embed = torch.nn.Linear(in_features=feature_size, out_features=pad_speech_size)
    with torch.no_grad():
        new_codec_embed.weight[:speech_token_size] = model.speech_embedding.weight
        new_codec_embed.weight[speech_token_size:] = 0
    model.llm.model.set_input_embeddings(new_codec_embed)

    model.llm.model.llm_embedding = model.llm_embedding
    logger.info(f"new_lm_head weight shape {new_lm_head.weight.shape}")
    logger.info(f"new_lm_head bias shape {new_lm_head.weight.shape}")

    logger.info("speech_embedding.weight")
    logger.info(new_codec_embed.weight)
    logger.info("embed_tokens.weight")
    logger.info(model.llm.model.text_embedding.weight)
    logger.info("llm_embedding.weight")
    logger.info(model.llm.model.llm_embedding.weight)
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = pad_speech_size
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = True
    model.llm.model.config.architectures = ["CosyVoice2ForCausalLM"]

    model.llm.model.save_pretrained(export_path)
    os.system(
        'sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g {}/config.json'.format(os.path.abspath(export_path)))

    tk_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",  # BPE/WordPiece 之一，若不存在会被忽略
        "merges.txt",  # 若是 BPE
        "added_tokens.json",  # 如有
    ]
    for fname in tk_files:
        src = os.path.join(pretrained_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(export_path, fname))

    with open(os.path.join(export_path, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    current_max_id = max(vocab.values())
    print(f"当前 vocab 最大 id: {current_max_id}")

    next_id = current_max_id + 1
    while next_id <= pad_vocab_size:
        # 添加占位 token，名称可随意
        vocab[f"<placeholder_{next_id}>"] = next_id
        next_id += 1

    # 保存回文件
    with open(os.path.join(export_path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"扩充完成，新 vocab 长度: {len(vocab)}")

    del model


def export_cosyvoice3_vllm(model_dir, device):
    export_path = os.path.join(model_dir, "vllm")
    if os.path.exists(export_path):
        return
    pretrained_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    os.makedirs(export_path, exist_ok=True)

    hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
    if not os.path.exists(hyper_yaml_path):
        raise ValueError('{} not found!'.format(hyper_yaml_path))
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f,
                                   overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    dtype = torch.bfloat16
    model = configs["llm"]
    model.load_state_dict(torch.load(f"{model_dir}/llm.pt", map_location=device), strict=True)
    model.to(device).eval()
    model.llm.model.to(device)
    model.llm.model.to(dtype)

    logger.info(model.llm.model.config)
    pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
    vocab_size = model.llm.model.config.vocab_size
    speech_token_size = model.speech_embedding.num_embeddings
    feature_size = model.speech_embedding.embedding_dim
    pad_speech_size = ((speech_token_size + pad_to - 1) // pad_to) * pad_to
    pad_vocab_size = ((vocab_size + speech_token_size + 3 + pad_to - 1) // pad_to) * pad_to
    logger.info(f"speech_embedding vocab_size {vocab_size}")
    logger.info(f"speech_token_size {speech_token_size}")
    logger.info(f"pad_speech_size {pad_speech_size}")
    logger.info(f"pad_vocab_size {pad_vocab_size}")
    logger.info(f"model.speech_embedding.weight shape {model.speech_embedding.weight.shape}")

    # lm_head
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=pad_speech_size, bias=False)
    logger.info(f"model.llm_decoder.weight shape {model.llm_decoder.weight.shape}")
    with torch.no_grad():
        new_lm_head.weight[:speech_token_size] = model.llm_decoder.weight
        new_lm_head.weight[speech_token_size:] = 0
    model.llm.model.lm_head = new_lm_head

    model.llm.model.text_embedding = model.llm.model.model.embed_tokens

    new_codec_embed = torch.nn.Linear(in_features=feature_size, out_features=pad_speech_size)
    with torch.no_grad():
        new_codec_embed.weight[:speech_token_size] = model.speech_embedding.weight
        new_codec_embed.weight[speech_token_size:] = 0
    model.llm.model.set_input_embeddings(new_codec_embed)

    logger.info(f"new_lm_head weight shape {new_lm_head.weight.shape}")
    logger.info(f"new_lm_head bias shape {new_lm_head.weight.shape}")

    logger.info(model.llm.model.text_embedding.weight)
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = pad_speech_size
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = True
    model.llm.model.config.architectures = ["CosyVoice3ForCausalLM"]

    model.llm.model.save_pretrained(export_path)
    os.system(
        'sed -i s@Qwen2ForCausalLM@CosyVoice3ForCausalLM@g {}/config.json'.format(os.path.abspath(export_path)))

    tk_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",  # BPE/WordPiece 之一，若不存在会被忽略
        "merges.txt",  # 若是 BPE
        "added_tokens.json",  # 如有
    ]
    for fname in tk_files:
        src = os.path.join(pretrained_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(export_path, fname))

    with open(os.path.join(export_path, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    current_max_id = max(vocab.values())
    print(f"当前 vocab 最大 id: {current_max_id}")

    next_id = current_max_id + 1
    while next_id <= pad_vocab_size:
        # 添加占位 token，名称可随意
        vocab[f"<placeholder_{next_id}>"] = next_id
        next_id += 1

    # 保存回文件
    with open(os.path.join(export_path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"扩充完成，新 vocab 长度: {len(vocab)}")

    del model


def auth_validate(app_id: bytes, nonce: bytes, signature: bytes):
    if not app_id or not nonce or not signature:
        return False
    app_id = app_id.decode()
    nonce = nonce.decode()
    signature = signature.decode()
    token = auth_info.get(app_id, None)
    if not token:
        return False

    md5_hash = hashlib.md5()
    input_string = app_id + nonce
    md5_hash.update(input_string.encode())
    digest = md5_hash.hexdigest()

    md5_hash = hashlib.md5()
    input_string = digest + token
    md5_hash.update(input_string.encode())

    true_signature = md5_hash.hexdigest()

    return true_signature == signature
