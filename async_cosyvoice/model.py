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

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.flow.flow_matching import EstimatorWrapper
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt

# 启用vllm V1版本
os.environ["VLLM_USE_V1"] = '1'
from vllm import  AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from vllm import ModelRegistry
from async_cosyvoice.config import ENGINE_ARGS, SAMPLING_PARAMS, ESTIMATOR_COUNT

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
         flow: CausalMaskedDiffWithXvec | torch.nn.Module,
         hift: HiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
    ):
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
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
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        # CausalMaskedDiffWithXvec has encoder, CausalMaskedDiffWithDiT uses pre_lookahead_layer instead
        if hasattr(self.flow, 'encoder') and self.flow.encoder is not None:
            self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
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

        # 与vllm中的模型保持一致 (CosyVoice3 token 编码方案)
        # - 基础语音 token: 0 ~ 6560 (6561个)
        # - 特殊 token: 6561 ~ 6760 (200个，包括 sos, eos, task, fill 等)
        self.speech_token_size = 6761  # CosyVoice3: 6561+200=6761
        self.base_speech_token_size = 6561  # 基础语音token数量
        self.vocab_size = 151936  # Qwen2 vocab_size

        # 新的token编码方案：使用 vocab_size 作为分界点，确保 text tokens 和 speech tokens 不重叠
        # - text tokens: 0 ~ vocab_size-1 (原始 Qwen2 tokenizer 值)
        # - speech tokens: vocab_size ~ vocab_size + speech_token_size - 1
        self.speech_token_offset = self.vocab_size  # 151936
        self.text_token_offset = 0  # 文本token不加偏移，直接使用原始值

        # CosyVoice3 的特殊 token (原始值: sos=6561, eos=6562, task=6563, fill=6564)
        # 加偏移后:
        self.sos_eos_token_id = self.speech_token_offset + self.base_speech_token_size + 0   # 158497
        self.eos_token_id = self.speech_token_offset + self.base_speech_token_size + 1       # 158498
        self.task_token_id = self.speech_token_offset + self.base_speech_token_size + 2      # 158499
        self.fill_token_id = self.speech_token_offset + self.base_speech_token_size + 3      # 158500
        self.zero_token_id = self.speech_token_offset + self.speech_token_size               # 158697

        # CosyVoice3: stop_token_ids 包含所有200个特殊token (索引 6561~6760)
        # 加偏移后: [151936+6561, ..., 151936+6760] = [158497, ..., 158696]
        self.stop_token_ids = list(range(self.speech_token_offset + self.base_speech_token_size,
                                         self.speech_token_offset + self.speech_token_size))

        # vllm 的推理任务需要在一个固定的事件循环中，因此启动一个后台线程专用于推理任务
        self.background_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

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
        logging.info(f'background_llm_inference started, request_id: {request_id}, prompt_len: {len(prompt_token_ids)}')
        logging.info(f'prompt_token_ids first 10: {prompt_token_ids[:10]}, last 10: {prompt_token_ids[-10:]}')
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or self.stop_token_ids
        logging.info(f'stop_token_ids count: {len(sampling_params.stop_token_ids)}, first: {sampling_params.stop_token_ids[0] if sampling_params.stop_token_ids else None}')
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        import sys
        logging.info(f'about to call llm_engine.generate, max_tokens: {sampling_params.max_tokens}')
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        logging.info(f'calling llm_engine.generate with prompt_token_ids len: {len(prompt_token_ids)}')
        try:
            output_count = 0
            async for output in self.llm_engine.generate(
                    {
                        "prompt_token_ids": prompt_token_ids,
                    },
                    sampling_params=sampling_params,
                    request_id=request_id or f"{time.time()}",
            ):
                output_count += 1
                logging.info(f'llm_engine output #{output_count}, finished: {output.finished}, tokens: {len(output.outputs[0].token_ids)}')
                out_queue.put((output.outputs[0], output.finished))
            logging.info(f'llm_engine.generate completed, total outputs: {output_count}')
        except Exception as e:
            logging.error(f'llm_engine.generate error: {e}', exc_info=True)
            raise;

    async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None):
        logging.info(f'llm_inference started, request_id: {request_id}')
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or self.stop_token_ids
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        logging.info(f'llm_inference calling generate, prompt_len: {len(prompt_token_ids)}')
        try:
            async for output in self.llm_engine.generate(
                    {
                        "prompt_token_ids": prompt_token_ids,
                    },
                    sampling_params=sampling_params,
                    request_id=request_id or f"{time.time()}",
            ):
                logging.debug(f'llm_inference got output, finished: {output.finished}')
                yield output.outputs[0]
        except Exception as e:
            logging.error(f'llm_engine.generate error: {e}', exc_info=True)
            raise

    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        logging.info(f'llm_job started, uuid: {uuid}')
        # text tokens 不加偏移 (text_token_offset=0)
        prompt_text = tensor_to_list(prompt_text + torch.tensor(self.text_token_offset))
        # speech tokens 加偏移，映射到词汇表末尾
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token + torch.tensor(self.speech_token_offset))
        logging.info(f'llm_job prompt_text len: {len(prompt_text)}, llm_prompt_speech_token len: {len(llm_prompt_speech_token)}')

        start_time = time.time()
        if isinstance(text, Union[Generator,AsyncGenerator]):
            if isinstance(text, Generator):
                text = AsyncWrapper(text)
            last_tokens = []
            prompt_token_ids = [self.sos_eos_token_id]
            text_tokens_cache = prompt_text
            async for this_text in text:
                this_text = tensor_to_list(this_text + torch.tensor(self.text_token_offset))
                # text need tokens
                text_tokens_cache += this_text
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
                    # 使用 self.fill_token_id (151442)
                    if (len(last_tokens) > 0 and last_tokens[-1] == self.fill_token_id) or len(prompt_token_ids) == 1:
                        if len(text_tokens_cache) >= self.mix_ratio[0]:
                            text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                            prompt_token_ids += text_tokens_temp
                            text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        else:
                            continue
                    # 注意：config.py 中设置了 output_kind=DELTA，所以 output.token_ids 已经是增量的
                    async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[self.fill_token_id]):
                        # DELTA 模式下，token_ids 已经是增量的，直接使用
                        new_tokens = list(output.token_ids)

                        if len(new_tokens) == 0:
                            continue

                        last_tokens = new_tokens
                        # 检查是否是 speech token (>= speech_token_offset + base_speech_token_size)
                        if last_tokens[-1] >= self.speech_token_offset + self.base_speech_token_size:
                            need_add_tokens = last_tokens[:-1]
                        else:
                            need_add_tokens = last_tokens
                        # 输出的 speech tokens 需要减去偏移，还原为原始值 (用于 tts_speech_token_dict)
                        decoded_tokens = [t - self.speech_token_offset if t >= self.speech_token_offset else t for t in need_add_tokens]
                        self.tts_speech_token_dict[uuid].extend(decoded_tokens)
                        # prompt_token_ids 保持编码后的值
                        prompt_token_ids.extend(need_add_tokens)

            prompt_token_ids += text_tokens_cache + [self.task_token_id]
            # 注意：config.py 中设置了 output_kind=DELTA，所以 output.token_ids 已经是增量的
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=self.stop_token_ids):
                # DELTA 模式下，token_ids 已经是增量的，直接使用
                new_tokens = list(output.token_ids)

                if len(new_tokens) == 0:
                    continue

                # 检查是否是 stop token (>= speech_token_offset + base_speech_token_size)
                if new_tokens[-1] >= self.speech_token_offset + self.base_speech_token_size:
                    need_add_tokens = new_tokens[:-1]
                else:
                    need_add_tokens = new_tokens
                # 输出的 speech tokens 需要减去偏移，还原为原始值
                need_add_tokens = [t - self.speech_token_offset if t >= self.speech_token_offset else t for t in need_add_tokens]
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)
        else:
            text = tensor_to_list(text + torch.tensor(self.text_token_offset))
            prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            max_tokens = len(text) * 20

            # 调试日志
            logging.info(f'[DEBUG] prompt structure: sos={self.sos_eos_token_id}, task={self.task_token_id}')
            logging.info(f'[DEBUG] prompt_text len={len(prompt_text)}, text len={len(text)}, speech_token len={len(llm_prompt_speech_token)}')
            logging.info(f'[DEBUG] total prompt_token_ids len={len(prompt_token_ids)}')
            logging.info(f'[DEBUG] speech_token_offset={self.speech_token_offset}, base_speech_token_size={self.base_speech_token_size}')

            # 检查 text token 是否与 speech token 范围重叠
            text_token_max = max(prompt_text + text) if (prompt_text + text) else 0
            text_token_min = min(prompt_text + text) if (prompt_text + text) else 0
            logging.info(f'[DEBUG] text token range: min={text_token_min}, max={text_token_max}')
            if text_token_max >= self.speech_token_offset:
                logging.warning(f'[WARNING] Text token {text_token_max} >= speech_token_offset {self.speech_token_offset}! Token ranges overlap!')

            # 注意：config.py 中设置了 output_kind=DELTA，所以 output.token_ids 已经是增量的
            output_count = 0

            # 添加调试：打印 prompt_token_ids 的关键部分
            logging.info(f'[DEBUG] prompt_token_ids first 10: {prompt_token_ids[:10]}')
            logging.info(f'[DEBUG] prompt_token_ids last 10: {prompt_token_ids[-10:]}')
            logging.info(f'[DEBUG] llm_prompt_speech_token first 5: {llm_prompt_speech_token[:5]}')

            async for output in self.llm_inference(
                    prompt_token_ids,
                    request_id=uuid,
                    stop_token_ids=self.stop_token_ids,
                    max_tokens=max_tokens,
            ):
                output_count += 1
                # DELTA 模式下，token_ids 已经是增量的，直接使用
                new_tokens = list(output.token_ids)

                if output_count <= 5 or output_count % 50 == 0:
                    logging.info(f'[DEBUG] output #{output_count}: raw tokens={new_tokens} (offset={self.speech_token_offset})')

                if len(new_tokens) == 0:
                    continue

                # 检查是否是 stop token (>= speech_token_offset + base_speech_token_size)
                is_stop = new_tokens[-1] >= self.speech_token_offset + self.base_speech_token_size
                if is_stop:
                    need_add_tokens = new_tokens[:-1]
                else:
                    need_add_tokens = new_tokens

                # 输出的 speech tokens 需要减去偏移，还原为原始值
                decoded_tokens = [t - self.speech_token_offset if t >= self.speech_token_offset else t for t in need_add_tokens]
                if output_count <= 5:
                    logging.info(f'[DEBUG] decoded tokens={decoded_tokens}')
                self.tts_speech_token_dict[uuid].extend(decoded_tokens)

            logging.info(f'[DEBUG] total outputs: {output_count}')

        self.llm_end_dict[uuid] = True
        tokens = self.tts_speech_token_dict[uuid]
        logging.info(f'llm job done, generated {len(tokens):>4} tokens, time cost: {time.time() - start_time:.3f}s')
        if len(tokens) > 0:
            logging.info(f'speech_tokens range: min={min(tokens)}, max={max(tokens)}, first 10: {tokens[:10]}')
        logging.debug(f'speech_tokens: len: {len(tokens)}  data: {tokens}')
        # 记录 prompt_token_ids self.tts_speech_token_dict[uuid] 数据用于后续的训练，与flow推理测试


    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        torch.cuda.current_stream().synchronize() # 将当前流进行同步了再处理后续逻辑
        stream = self.stream_pool.get()
        with torch.cuda.stream(stream):
            # 根据 fp16 设置转换数据类型
            dtype = torch.float16 if self.fp16 else torch.float32
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device, dtype=dtype),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device, dtype=dtype),
                                             streaming=True,
                                             finalize=finalize)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]

            # 检查 hift 类型，CausalHiFTGenerator 使用不同的接口
            from cosyvoice.hifigan.generator import CausalHiFTGenerator
            is_causal_hift = isinstance(self.hift, CausalHiFTGenerator)

            if is_causal_hift:
                # CausalHiFTGenerator: 使用 finalize 参数，不需要 cache
                if speed != 1.0 and finalize:
                    tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
                tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            else:
                # HiFTGenerator: 使用 cache_source 参数
                # append hift cache
                if self.hift_cache_dict[uuid] is not None:
                    hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
                    tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                else:
                    hift_cache_source = torch.zeros(1, 1, 0)
                # keep overlap mel and hift cache
                if finalize is False:
                    tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                    if self.hift_cache_dict[uuid] is not None:
                        tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
                    self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                                  'source': tts_source[:, :, -self.source_cache_len:],
                                                  'speech': tts_speech[:, -self.source_cache_len:]}
                    tts_speech = tts_speech[:, :-self.source_cache_len]
                else:
                    if speed != 1.0:
                        assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                        tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
                    tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                    if self.hift_cache_dict[uuid] is not None:
                        tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            torch.cuda.synchronize(torch.cuda.current_stream())
            self.stream_pool.put(stream)
            return tts_speech

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
                    this_tts_speech = await loop.run_in_executor(self.thread_executor,
                        self.token2wav,
                        this_tts_speech_token,
                            flow_prompt_speech_token,
                            prompt_speech_feat,
                            flow_embedding,
                            this_uuid,
                            token_offset,
                            False
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
            this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                         self.token2wav,
                                                         this_tts_speech_token,
                                                         flow_prompt_speech_token,
                                                         prompt_speech_feat,
                                                         flow_embedding,
                                                         this_uuid,
                                                         token_offset,
                                                         True,
                                                         )
            if this_tts_speech.shape[1] == 0:
                logging.debug(f'no tts_speech_token shape: {this_tts_speech.shape}, data: {this_tts_speech}')
                return
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            await llm_task
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            loop = asyncio.get_event_loop()
            this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                         self.token2wav,
                                                         this_tts_speech_token,
                                                         flow_prompt_speech_token,
                                                         prompt_speech_feat,
                                                         flow_embedding,
                                                         this_uuid,
                                                         0,
                                                         True,
                                                         speed,
                                                         )
            yield {'tts_speech': this_tts_speech.cpu()}
        async with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
