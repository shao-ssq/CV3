# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
独立的 Token2Wav 模型，用于分布式部署
只加载 flow + hift，不加载 LLM
"""
import os
import logging
from typing import Dict, Any, Optional

import torch
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.flow.flow_matching import EstimatorWrapper
from cosyvoice.utils.file_utils import convert_onnx_to_trt


class Token2WavModel:
    """独立的 Token2Wav 模型"""

    def __init__(
        self,
        model_dir: str,
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        device: str = None,
        estimator_count: int = 4
    ):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # 加载配置
        with open(f'{model_dir}/cosyvoice3.yaml', 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'model_path': model_dir})

        self.sample_rate = configs['sample_rate']
        self.flow = configs['flow']
        self.hift = configs['hift']

        # 加载权重
        self._load_weights(model_dir)

        # 设置 fp16
        self.flow.fp16 = fp16
        if fp16:
            self.flow.half()

        # 设置静态 chunk size
        self.flow.pre_lookahead_layer.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = (
            2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        )

        # 加载优化模型
        if load_jit:
            self._load_jit(model_dir)
        if load_trt:
            self._load_trt(model_dir, estimator_count)

        # 会话缓存
        self.hift_cache_dict: Dict[str, Any] = {}

        logging.info(f'Token2WavModel initialized on {self.device}, fp16={fp16}')
        del configs

    def _load_weights(self, model_dir: str):
        """加载模型权重"""
        flow_state = torch.load(
            f'{model_dir}/flow.pt',
            weights_only=True,
            map_location=self.device
        )
        self.flow.load_state_dict(flow_state, strict=True)
        self.flow.to(self.device).eval()

        hift_state = torch.load(
            f'{model_dir}/hift.pt',
            weights_only=True,
            map_location=self.device
        )
        hift_state = {k.replace('generator.', ''): v for k, v in hift_state.items()}
        self.hift.load_state_dict(hift_state, strict=True)
        self.hift.to(self.device).eval()

    def _load_jit(self, model_dir: str):
        """加载 JIT 编译的 encoder"""
        suffix = 'fp16' if self.fp16 else 'fp32'
        jit_path = f'{model_dir}/flow.encoder.{suffix}.zip'
        if os.path.exists(jit_path):
            self.flow.encoder = torch.jit.load(jit_path, map_location=self.device)
            logging.info(f'Loaded JIT encoder from {jit_path}')

    def _load_trt(self, model_dir: str, estimator_count: int):
        """加载 TensorRT 模型"""
        suffix = 'fp16' if self.fp16 else 'fp32'
        trt_path = f'{model_dir}/flow.decoder.estimator.{suffix}.mygpu.plan'
        onnx_path = f'{model_dir}/flow.decoder.estimator.fp32.onnx'

        if not os.path.exists(trt_path):
            logging.info(f'Converting ONNX to TensorRT: {trt_path}')
            convert_onnx_to_trt(trt_path, onnx_path, self.fp16)

        import tensorrt as trt
        with open(trt_path, 'rb') as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())

        del self.flow.decoder.estimator
        self.flow.decoder.estimator = EstimatorWrapper(engine, estimator_count=estimator_count)
        logging.info(f'Loaded TensorRT estimator with {estimator_count} instances')

    @torch.inference_mode()
    def token2wav(
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
        将 speech tokens 转换为音频波形

        Args:
            token: 语音 tokens, shape (1, seq_len)
            prompt_token: 提示 tokens, shape (1, prompt_len)
            prompt_feat: 提示 mel 特征, shape (1, mel_len, 80)
            embedding: 说话人嵌入, shape (1, 192)
            token_offset: 流式推理的 token 偏移
            session_id: 会话 ID
            stream: 是否流式模式
            finalize: 是否是最后一个 chunk
            speed: 语速控制

        Returns:
            音频波形 tensor, shape (1, samples)
        """
        with torch.amp.autocast('cuda', enabled=self.fp16):
            # flow 推理: tokens → mel
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=stream,
                finalize=finalize
            )

            # 截取有效部分
            tts_mel = tts_mel[:, :, int(token_offset * self.flow.token_mel_ratio):]

            # 处理缓存
            if session_id not in self.hift_cache_dict:
                self.hift_cache_dict[session_id] = None

            if self.hift_cache_dict[session_id] is not None:
                hift_cache_mel = self.hift_cache_dict[session_id]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[session_id]['mel'] = tts_mel
            else:
                self.hift_cache_dict[session_id] = {'mel': tts_mel, 'speech_offset': 0}

            # 语速调整 (仅非流式)
            if speed != 1.0:
                assert token_offset == 0 and finalize, 'speed change only supports non-stream mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

            # hift 推理: mel → wav
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[session_id]['speech_offset']:]
            self.hift_cache_dict[session_id]['speech_offset'] += tts_speech.shape[1]

        return tts_speech

    def clear_session(self, session_id: str):
        """清理会话缓存"""
        if session_id in self.hift_cache_dict:
            del self.hift_cache_dict[session_id]
            logging.debug(f'Cleared session cache: {session_id}')

    def get_session_count(self) -> int:
        """获取当前活跃会话数"""
        return len(self.hift_cache_dict)
