"""
Token2Wav gRPC 客户端
用于分布式部署时调用远程 Token2Wav 服务
"""
import os
import sys
import logging
from typing import List, Dict, Any
import hashlib

import torch
import numpy as np
import grpc
from grpc import aio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import cosyvoice_pb2
import cosyvoice_pb2_grpc


def serialize_tensor(tensor: torch.Tensor) -> tuple:
    """序列化 tensor 为 bytes 和 meta"""
    if tensor.numel() == 0:
        return b'', cosyvoice_pb2.TensorMeta(shape=[], dtype='float32')

    arr = tensor.cpu().numpy()
    dtype_str = str(arr.dtype).replace('numpy.', '').replace('torch.', '')
    if 'float32' in dtype_str:
        dtype_str = 'float32'
    elif 'float16' in dtype_str:
        dtype_str = 'float16'
    elif 'int32' in dtype_str:
        dtype_str = 'int32'
    elif 'int64' in dtype_str:
        dtype_str = 'int64'
    else:
        arr = arr.astype(np.float32)
        dtype_str = 'float32'

    return arr.tobytes(), cosyvoice_pb2.TensorMeta(shape=list(arr.shape), dtype=dtype_str)


def deserialize_audio(data: bytes, format: str = 'raw_float32') -> torch.Tensor:
    """反序列化音频数据"""
    if format == 'pcm_int16':
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        arr = np.frombuffer(data, dtype=np.float32)
    return torch.from_numpy(arr.copy()).unsqueeze(0)


class Token2WavClient:
    """Token2Wav 远程调用客户端"""

    def __init__(self, services: List[Dict[str, Any]], strategy: str = 'session_sticky', timeout_ms: int = 30000):
        """
        Args:
            services: 服务列表 [{"host": "localhost", "port": 50001}, ...]
            strategy: 负载均衡策略 (round_robin, session_sticky)
            timeout_ms: 超时时间 (毫秒)
        """
        self.services = services
        self.strategy = strategy
        self.timeout = timeout_ms / 1000.0
        self.channels: Dict[str, aio.Channel] = {}
        self.stubs: Dict[str, cosyvoice_pb2_grpc.Token2WavServiceStub] = {}
        self.round_robin_idx = 0

        logging.info(f'Token2WavClient initialized with {len(services)} services, strategy={strategy}')

    def _get_service_key(self, host: str, port: int) -> str:
        return f"{host}:{port}"

    async def _get_stub(self, service: Dict[str, Any]) -> cosyvoice_pb2_grpc.Token2WavServiceStub:
        """获取或创建 gRPC stub"""
        key = self._get_service_key(service['host'], service['port'])
        if key not in self.stubs:
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ]
            channel = aio.insecure_channel(key, options=options)
            self.channels[key] = channel
            self.stubs[key] = cosyvoice_pb2_grpc.Token2WavServiceStub(channel)
        return self.stubs[key]

    def _select_service(self, session_id: str = None) -> Dict[str, Any]:
        """根据策略选择服务"""
        if len(self.services) == 1:
            return self.services[0]

        if self.strategy == 'session_sticky' and session_id:
            # 基于 session_id 的一致性哈希
            hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            idx = hash_val % len(self.services)
            return self.services[idx]
        else:
            # Round Robin
            service = self.services[self.round_robin_idx]
            self.round_robin_idx = (self.round_robin_idx + 1) % len(self.services)
            return service

    async def token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        token_offset: int,
        session_id: str,
        stream: bool = False,
        finalize: bool = False,
        speed: float = 1.0
    ) -> torch.Tensor:
        """
        调用远程 Token2Wav 服务

        Args:
            token: 语音 tokens, shape (1, seq_len)
            prompt_token: 提示 tokens
            prompt_feat: 提示 mel 特征
            embedding: 说话人嵌入
            token_offset: 流式偏移
            session_id: 会话 ID
            stream: 流式模式
            finalize: 是否结束
            speed: 语速

        Returns:
            音频波形 tensor
        """
        service = self._select_service(session_id)
        stub = await self._get_stub(service)

        # 序列化 tensors
        prompt_token_bytes, prompt_token_meta = serialize_tensor(prompt_token)
        prompt_feat_bytes, prompt_feat_meta = serialize_tensor(prompt_feat)
        embedding_bytes, embedding_meta = serialize_tensor(embedding)

        # 构建请求
        request = cosyvoice_pb2.Token2WavRequest(
            session_id=session_id,
            speech_tokens=token.view(-1).tolist(),
            prompt_token=prompt_token_bytes,
            prompt_token_meta=prompt_token_meta,
            prompt_feat=prompt_feat_bytes,
            prompt_feat_meta=prompt_feat_meta,
            embedding=embedding_bytes,
            embedding_meta=embedding_meta,
            token_offset=token_offset,
            stream=stream,
            finalize=finalize,
            speed=speed
        )

        try:
            response = await stub.Convert(request, timeout=self.timeout)
            audio = deserialize_audio(response.audio_data, response.format)
            return audio
        except grpc.RpcError as e:
            logging.error(f'Token2Wav RPC failed: {e.code()}: {e.details()}')
            raise

    async def close(self):
        """关闭所有连接"""
        for channel in self.channels.values():
            await channel.close()
        self.channels.clear()
        self.stubs.clear()
