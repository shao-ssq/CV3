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
import asyncio
import logging
import os
import queue
import threading
from typing import AsyncGenerator, Generator, List, Union
import torch
import numpy as np
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from concurrent.futures import ThreadPoolExecutor

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec, CausalMaskedDiffWithDiT
from cosyvoice.flow.flow_matching import EstimatorWrapper
from cosyvoice.hifigan.generator import HiFTGenerator, CausalHiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt
from vllm import  AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from vllm import ModelRegistry
from async_cosyvoice.config import (
    ENGINE_ARGS, SAMPLING_PARAMS, ESTIMATOR_COUNT,
    DISTRIBUTED_MODE, TOKEN2WAV_SERVICES, LOAD_BALANCE_STRATEGY, TOKEN2WAV_TIMEOUT_MS
)

from async_cosyvoice.vllm_use_cosyvoice2_model import CosyVoice2Model as CosyVoice2LLM
ModelRegistry.register_model("CosyVoice2Model", CosyVoice2LLM)

class AsyncWrapper:
    """将一个同步生成器包装为异步生成器"""
    def __init__(self, obj):
        self.obj = obj

    async def __aiter__(self):
        for item in self.obj:
            yield item

def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()


class CosyVoice2Model:

    def __init__(self,
         model_dir: str,
         flow: CausalMaskedDiffWithDiT | torch.nn.Module,
         hift: CausalHiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
    ):
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
            skip_tokenizer_init=True,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)
        self.thread_count = 10
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_count)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.stream_pool = queue.Queue()
        for _ in range(self.thread_count):
            stream = torch.cuda.Stream(self.device)
            self.stream_pool.put(stream)

        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

        # 设置 fp16 模式
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()

        self.token_hop_len = 2 * self.flow.input_frame_rate
        # 设置静态 chunk size，有利于 CUDA JIT 编译优化
        self.flow.pre_lookahead_layer.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()

        self.mix_ratio = mix_ratio or [5, 15]

        self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致 (简化的 CosyVoice3 token 编码方案)
        # - speech tokens (包括特殊token): 0 ~ 6760
        #   - 基础语音 token: 0 ~ 6560 (6561个)
        #   - 特殊 token: 6561 ~ 6760 (200个，包括 sos, eos, task, fill 等)
        # - text tokens: >= 6761 (原始 Qwen2 tokenizer 值 + 6761)
        speech_token_size = int(6561)
        self.speech_token_num = speech_token_size + 200  # 6761
        self.base_speech_token_size = speech_token_size  # 6561

        # llm_token_id_delta 作为分界点
        self.llm_token_id_delta = self.speech_token_num  # 6761

        # 特殊 token IDs (在 speech token 范围内)
        self.sos_eos_token_id = self.base_speech_token_size + 0  # 6561
        self.eos_token_id = self.base_speech_token_size + 1      # 6562
        self.task_token_id = self.base_speech_token_size + 2     # 6563
        self.fill_token_id = self.base_speech_token_size + 3     # 6564

        # stop_token_ids 包含所有200个特殊token (6561~6760)
        self.stop_token_ids = [i for i in range(speech_token_size, speech_token_size + 200)]

        # vllm 的推理任务需要在一个固定的事件循环中，因此启动一个后台线程专用于推理任务
        self.background_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        # 分布式模式：初始化远程 Token2Wav 客户端
        self.distributed_mode = DISTRIBUTED_MODE
        self.token2wav_client = None
        if self.distributed_mode and TOKEN2WAV_SERVICES:
            from async_cosyvoice.runtime.async_grpc.token2wav_client import Token2WavClient
            self.token2wav_client = Token2WavClient(
                services=TOKEN2WAV_SERVICES,
                strategy=LOAD_BALANCE_STRATEGY,
                timeout_ms=TOKEN2WAV_TIMEOUT_MS
            )
            logging.info(f'Distributed mode enabled, {len(TOKEN2WAV_SERVICES)} Token2Wav services configured')

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, weights_only=True, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, weights_only=True, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = EstimatorWrapper(self.flow.decoder.estimator_engine, estimator_count=ESTIMATOR_COUNT)

    async def background_llm_inference(self, out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens):
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or [6561, 6563]
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        async for output in self.llm_engine.generate(
                {
                    "prompt_token_ids": prompt_token_ids,
                },
                sampling_params=sampling_params,
                request_id=request_id or f"{time.time()}",
        ):
            out_queue.put((output.outputs[0], output.finished))

    async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None, min_tokens=None):
        logging.info(f'llm_inference started, request_id: {request_id}')
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or self.stop_token_ids
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        if min_tokens:
            sampling_params.min_tokens = min_tokens

        logging.info(f'llm_inference calling generate, prompt_len: {len(prompt_token_ids)}')
        try:
            async for output in self.llm_engine.generate(
                    {
                        "prompt_token_ids": prompt_token_ids,
                    },
                    sampling_params=sampling_params,
                    request_id=request_id or f"{time.time()}",
            ):
                yield output.outputs[0]
        except Exception as e:
            logging.error(f'llm_engine.generate error: {e}', exc_info=True)
            raise

    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        prompt_text = tensor_to_list(prompt_text + torch.tensor(self.llm_token_id_delta))
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)

        # 检查输入的 prompt_speech_token 范围
        if llm_prompt_speech_token:
            min_ps = min(llm_prompt_speech_token)
            max_ps = max(llm_prompt_speech_token)
            logging.info(f'[INPUT CHECK] llm_prompt_speech_token: len={len(llm_prompt_speech_token)}, min={min_ps}, max={max_ps}')
            if max_ps >= self.base_speech_token_size:
                logging.warning(f'[INPUT CHECK] WARNING: prompt_speech_token contains values >= {self.base_speech_token_size}!')

        start_time = time.time()
        if isinstance(text, Union[Generator,AsyncGenerator]):
            if isinstance(text, Generator):
                text = AsyncWrapper(text)
            last_tokens = []
            prompt_token_ids = [self.sos_eos_token_id]
            text_tokens_cache = prompt_text
            total_text_tokens = len(prompt_text)  # 跟踪总文本token数
            async for this_text in text:
                this_text = tensor_to_list(this_text + torch.tensor(self.llm_token_id_delta))
                # text need tokens
                text_tokens_cache += this_text
                total_text_tokens += len(this_text)  # 累计文本token数
                while len(llm_prompt_speech_token) != 0:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_input_token = text_tokens_cache[:self.mix_ratio[0]]
                        speech_input_token = llm_prompt_speech_token[:self.mix_ratio[1]]
                        prompt_token_ids += text_input_token + speech_input_token
                        # reset the last cache
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        llm_prompt_speech_token = llm_prompt_speech_token[self.mix_ratio[1]:]
                    else:
                        break
                if len(llm_prompt_speech_token) == 0:
                    if (len(last_tokens) > 0 and last_tokens[-1] == self.fill_token_id) or len(prompt_token_ids) == 1:
                        if len(text_tokens_cache) >= self.mix_ratio[0]:
                            text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                            prompt_token_ids += text_tokens_temp
                            text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        else:
                            continue
                    async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[self.fill_token_id]):
                        new_tokens = list(output.token_ids)

                        if len(new_tokens) == 0:
                            continue

                        last_tokens = new_tokens
                        # 检查是否是特殊 token (>= base_speech_token_size)
                        if last_tokens[-1] >= self.base_speech_token_size:
                            need_add_tokens = last_tokens[:-1]
                        else:
                            need_add_tokens = last_tokens
                        # speech tokens 直接使用，不需要减去偏移
                        self.tts_speech_token_dict[uuid].extend(need_add_tokens)
                        prompt_token_ids.extend(need_add_tokens)

            prompt_token_ids += text_tokens_cache + [self.task_token_id]
            # 计算已生成的token数，用于确定还需要生成多少
            already_generated = len(self.tts_speech_token_dict[uuid])
            expected_total = total_text_tokens * 2  # 预期总token数
            remaining_min = max(0, expected_total - already_generated)  # 还需要生成的最小数量
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid,
                                                   stop_token_ids=self.stop_token_ids,
                                                   min_tokens=remaining_min,
                                                   max_tokens=total_text_tokens * 20):
                new_tokens = list(output.token_ids)

                if len(new_tokens) == 0:
                    continue

                # 检查是否是 stop token (>= base_speech_token_size)
                if new_tokens[-1] >= self.base_speech_token_size:
                    need_add_tokens = new_tokens[:-1]
                else:
                    need_add_tokens = new_tokens
                # speech tokens 直接使用，不需要减去偏移
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)
        else:
            text = tensor_to_list(text + torch.tensor(self.llm_token_id_delta))
            prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            min_tokens = len(text) * 2
            max_tokens = len(text) * 20

            sampling_params = SamplingParams(
                **SAMPLING_PARAMS,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                stop_token_ids=self.stop_token_ids,
            )

            async for output in self.llm_engine.generate(
                    {
                        "prompt_token_ids": prompt_token_ids,
                    },
                    sampling_params=sampling_params,
                    request_id=uuid,
            ):
                output = output.outputs[0]
                if output.token_ids[-1] in self.stop_token_ids:
                    need_add_tokens = output.token_ids[:-1]
                else:
                    need_add_tokens = output.token_ids
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)

        self.llm_end_dict[uuid] = True
        logging.info(
            f'llm job done, generated {len(self.tts_speech_token_dict[uuid]):>4} tokens, time cost: {time.time() - start_time:.3f}s')
        logging.debug(
            f'speech_tokens: len: {len(self.tts_speech_token_dict[uuid])}  data: {self.tts_speech_token_dict[uuid]}')
        # 记录 prompt_token_ids self.tts_speech_token_dict[uuid] 数据用于后续的训练，与flow推理测试

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False,
                  speed=1.0):
        with torch.amp.autocast('cuda', enabled=self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(
                                                 self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                                                           dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                                 self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
            tts_mel = tts_mel[:, :, int(token_offset * self.flow.token_mel_ratio):]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech

    async def _remote_token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid,
                                 stream=False, finalize=False, speed=1.0):
        """调用远程 Token2Wav 服务"""
        return await self.token2wav_client.token2wav(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            token_offset=token_offset,
            session_id=uuid,
            stream=stream,
            finalize=finalize,
            speed=speed
        )

    async def _dispatch_token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid,
                                   stream=False, finalize=False, speed=1.0):
        """根据模式分发 token2wav 调用"""
        if self.distributed_mode and self.token2wav_client:
            return await self._remote_token2wav(
                token, prompt_token, prompt_feat, embedding,
                token_offset, uuid, stream, finalize, speed
            )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_executor,
                self.token2wav,
                token, prompt_token, prompt_feat, embedding,
                token_offset, uuid, stream, finalize, speed
            )

    async def async_tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        async with self.lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
        # queue: asyncio.Queue[int|None] = asyncio.Queue()
        llm_task = asyncio.create_task(self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        tts_start_time = time.time()
        if stream is True:
            token_offset = 0
            peer_chunk_token_num = 15     # 设置初始的每个chunk处理语音token的数量
            await asyncio.sleep(0.05)
            loop = asyncio.get_event_loop()
            start_time = time.time()
            chunk_index = 0
            while True:
                if (pending_num:= len(self.tts_speech_token_dict[this_uuid]) - token_offset) >= (peer_chunk_token_num + self.flow.pre_lookahead_len):
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + peer_chunk_token_num + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = await self._dispatch_token2wav(
                        this_tts_speech_token,
                        flow_prompt_speech_token,
                        prompt_speech_feat,
                        flow_embedding,
                        token_offset,
                        this_uuid,
                        True  # stream=True for streaming inference
                    )
                    token_offset += peer_chunk_token_num
                    yield {'tts_speech': this_tts_speech.cpu()}
                    chunk_index += 1
                    cost_time = time.time() - start_time
                    # 动态增大 peer_chunk_token_num，以减少调用 token2wav 的次数
                    duration = token_offset/self.flow.input_frame_rate
                    if (multiples:= (duration - cost_time)/(cost_time/chunk_index)) > 4:
                        if self.llm_end_dict[this_uuid] is True:   # 直接一次性推理 token2wav 返回剩余的语音
                            break
                        else:
                            # 有较多的计算时间，还可以等待llm生成更多的token，用于下次 token2wav 推理
                            peer_chunk_token_num = (pending_num // 15 + 1) * 15
                    elif multiples > 2:
                        peer_chunk_token_num = (pending_num // 15) * 15
                    logging.debug(f'the multiples: {multiples:.2f}, next chunk_token_num: {peer_chunk_token_num}')
                else:
                    if self.llm_end_dict[this_uuid] is True:
                        break
                    else:
                        await asyncio.sleep(0.02)
            await llm_task
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = await self._dispatch_token2wav(
                this_tts_speech_token,
                flow_prompt_speech_token,
                prompt_speech_feat,
                flow_embedding,
                token_offset,
                this_uuid,
                True,  # stream=True for streaming inference
                True   # finalize=True
            )
            if this_tts_speech.shape[1] == 0:
                logging.debug(f'no tts_speech_token shape: {this_tts_speech.shape}, data: {this_tts_speech}')
                return
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            await llm_task
            logging.info(f'[PERF] non-stream llm done, wait_time: {(time.time()-tts_start_time)*1000:.1f}ms, tokens: {len(self.tts_speech_token_dict[this_uuid])}')
            t_before_token2wav = time.time()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = await self._dispatch_token2wav(
                this_tts_speech_token,
                flow_prompt_speech_token,
                prompt_speech_feat,
                flow_embedding,
                0,
                this_uuid,
                False,
                True,  # finalize=True
                speed
            )
            logging.info(f'[PERF] non-stream token2wav done: {(time.time()-t_before_token2wav)*1000:.1f}ms, total: {(time.time()-tts_start_time)*1000:.1f}ms')
            yield {'tts_speech': this_tts_speech.cpu()}
        async with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
