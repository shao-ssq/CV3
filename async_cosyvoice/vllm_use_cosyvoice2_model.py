# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Set, Tuple, Union, Iterator, overload, TypedDict, Mapping, Any
from typing_extensions import TypeVar

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig, T
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model

from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings

logger = init_logger(__name__)

IGNORE_ID = -1


class CosyVoice2Model(nn.Module, SupportsLoRA, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.llm_input_size = 896
        self.llm_output_size = 896

        self.speech_token_size = 6561+200  # CosyVoice3: 6761, CosyVoice2: 6561+3
        self.llm_token_size = config.vocab_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2


        self.allow_patterns_overrides = ["llm.*"]
        self.llm_embedding = torch.nn.Embedding(2, self.llm_input_size)
        self.model = Qwen2Model(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "model"))

        # 注意：原始 CosyVoice3LM 是 bias=False
        self.llm_decoder = ParallelLMHead(self.speech_token_size,
                                      self.llm_output_size,
                                      bias=False,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(
                                          prefix, "llm_decoder"))
        self.logits_processor = LogitsProcessor(self.speech_token_size)

        # length_normalized_loss: bool = True,
        # lsm_weight: float = 0.0,
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=self.speech_token_size,
        #     padding_idx=IGNORE_ID,
        #     smoothing=lsm_weight,
        #     normalize_length=length_normalized_loss,
        # )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(self.speech_token_size, self.llm_input_size)

        # 4. sampling method
        ## use vllm sampling method
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.mix_ratio: List[int] = [5, 15]

        # CosyVoice3 token 编码方案：
        # - 基础语音 token: 0 ~ 6560 (6561个)
        # - 特殊 token: 6561 ~ 6760 (200个，包括 sos, eos, task, fill 等)
        #
        # 新的token编码方案：使用 vocab_size 作为分界点，确保 text tokens 和 speech tokens 不重叠
        # - text tokens: 0 ~ vocab_size-1 (原始 Qwen2 tokenizer 值)
        # - speech tokens: vocab_size ~ vocab_size + speech_token_size - 1
        # 注意：vLLM processor.py 中的 token 验证已被禁用，以支持扩展的 token ID 范围
        self.base_speech_token_size = 6561  # 基础语音token数量
        self.speech_token_offset = config.vocab_size  # 151936，使用 vocab_size 作为偏移

        # CosyVoice3 的特殊 token (使用 speech_embedding)
        # 原始值: sos=6561, eos=6562, task=6563, fill=6564
        # 加偏移后:
        self.sos_eos_token_id = self.speech_token_offset + self.base_speech_token_size + 0   # 151936 + 6561 = 158497
        self.eos_token_id = self.speech_token_offset + self.base_speech_token_size + 1       # 158498
        self.task_token_id = self.speech_token_offset + self.base_speech_token_size + 2      # 158499
        self.fill_token_id = self.speech_token_offset + self.base_speech_token_size + 3      # 158500

        # zero_token 用于填充
        self.zero_token_id = self.speech_token_offset + self.speech_token_size  # 158697

        self.zero_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, self.llm_input_size),
            dtype=self.speech_embedding.weight.dtype,
            device=self.speech_embedding.weight.device
        )
        self.inputs_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, self.llm_input_size),
            dtype=self.speech_embedding.weight.dtype,
            device=self.speech_embedding.weight.device,
        )

    def get_sos_eos_emb(self):
        # CosyVoice3: sos = 6561，使用 speech_embedding
        return self.speech_embedding.weight[self.base_speech_token_size + 0].reshape(1, 1, -1)

    def get_task_id_emb(self):
        # CosyVoice3: task = 6563，使用 speech_embedding
        return self.speech_embedding.weight[self.base_speech_token_size + 2].reshape(1, 1, -1)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[T] = None,
    ) -> torch.Tensor:
        """
        CosyVoice3 token 编码方案（使用 vocab_size 作为分界点，确保不重叠）：
        - text tokens: 0 ~ vocab_size-1 (原始 Qwen2 tokenizer 值，使用 model.get_input_embeddings)
        - speech tokens (包括特殊token): vocab_size ~ vocab_size+speech_token_size-1
          即 151936 ~ 158696 (减去偏移后使用 speech_embedding，索引 0~6760)
        - zero_token: 用于填充，嵌入为零向量
        """
        # 获取 input_ids 的原始形状
        input_shape = input_ids.shape
        flat_input_ids = input_ids.view(-1)

        inputs_embeds = self.inputs_embed_buffer[:flat_input_ids.shape[0]]
        inputs_embeds.zero_()

        # 创建各类型 token 的掩码
        # text tokens: < speech_token_offset (< 151936)
        text_mask = flat_input_ids < self.speech_token_offset
        # speech tokens (包括 sos, eos, task, fill 等特殊token): speech_token_offset ~ speech_token_offset+speech_token_size-1
        # 即 151936 ~ 158696
        speech_max_id = self.speech_token_offset + self.speech_token_size
        speech_mask = (flat_input_ids >= self.speech_token_offset) & (flat_input_ids < speech_max_id)
        # zero token: 特殊填充token
        zero_mask = flat_input_ids == self.zero_token_id

        # 处理 text tokens
        if text_mask.any():
            text_token_ids = flat_input_ids[text_mask]
            inputs_embeds[text_mask] = self.model.get_input_embeddings(text_token_ids)

        # 处理 speech tokens (减去偏移得到原始 speech token id: 0~6760)
        # 这包括基础语音token (0~6560) 和特殊token (6561~6760)
        if speech_mask.any():
            speech_token_ids = flat_input_ids[speech_mask] - self.speech_token_offset
            inputs_embeds[speech_mask] = self.speech_embedding(speech_token_ids)

        # 处理 zero token (填充用)
        if zero_mask.any():
            inputs_embeds[zero_mask] = self.zero_embed_buffer[:zero_mask.sum()]

        inputs_embeds = inputs_embeds.view(*input_shape, self.llm_input_size)

        # 合并多模态嵌入（如果有）
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.audio_token_index
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        logger.debug(f"CosyVoice2Model.forward called with input_ids shape: {input_ids.shape}")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # llm_decoder 输出 speech_token_size (6761) 个类别的 logits
        speech_logits = self.logits_processor(self.llm_decoder, hidden_states,
                                              sampling_metadata)
        if speech_logits is None:
            return None

        # 创建完整的 logits tensor，大小需要容纳所有 speech tokens
        # speech tokens 在 speech_token_offset ~ speech_token_offset + speech_token_size - 1
        # 即 151936 ~ 158696，所以需要 158697 大小的 tensor
        batch_size = speech_logits.shape[0]
        total_size = self.speech_token_offset + self.speech_token_size  # 151936 + 6761 = 158697
        full_logits = torch.full(
            (batch_size, total_size),
            float('-inf'),
            dtype=speech_logits.dtype,
            device=speech_logits.device
        )
        # 将 speech logits 放到正确的位置 (speech_token_offset ~ speech_token_offset + speech_token_size)
        full_logits[:, self.speech_token_offset:self.speech_token_offset + self.speech_token_size] = speech_logits
        return full_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    @staticmethod
    def convert_weights(weights: Iterable[Tuple[str, torch.Tensor]]) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, param in weights:
            # 处理 Qwen2Model 核心参数: llm.model.model.* -> model.*
            if name.startswith("llm.model.model."):
                name = name.replace("llm.model.model.", "model.")
                yield name, param
            # 处理 speech_embedding (顶层，无 llm. 前缀)
            elif name.startswith("speech_embedding."):
                yield name, param
            # 处理 llm_decoder (顶层，无 llm. 前缀)
            elif name.startswith("llm_decoder."):
                yield name, param
            # 跳过其他权重
            else:
                continue

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.convert_weights(weights)
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
