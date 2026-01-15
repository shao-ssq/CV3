import os
import time
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Generator, AsyncGenerator

import torch
import torch.nn.functional as F
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from async_cosyvoice.config import ENGINE_ARGS, SAMPLING_PARAMS
from async_cosyvoice.model import CosyVoice2Model, tensor_to_list
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.hifigan.generator import CausalHiFTGenerator


class CosyVoice3Model(CosyVoice2Model):
    def __init__(self,
                 model_dir: str,
                 flow: CausalMaskedDiffWithDiT | torch.nn.Module,
                 hift: CausalHiFTGenerator | torch.nn.Module,
                 fp16: bool = False):
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
            skip_tokenizer_init=True,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine | None = None if os.getenv("NOT_USE_VLLM",
                                                                   "") else AsyncLLMEngine.from_engine_args(engine_args)
        self.thread_count = 10
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_count)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # rtf and decoding related
        self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致
        speech_token_size = int(6561)
        self.speech_token_size = speech_token_size
        self.llm_token_size = 151936

        self.sos_token_id = speech_token_size + 0
        self.eos_token_id = speech_token_size + 1
        self.task_token_id = speech_token_size + 2
        self.fill_token_id = speech_token_size + 3
        self.speech_token_num = speech_token_size + 200

        self.zero_token_id = self.task_token_id + 1

        # vllm 的推理任务需要在一个固定的事件循环中，因此启动一个后台线程专用于推理任务
        self.background_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        self.stop_token_ids = [i for i in range(speech_token_size, speech_token_size + 200)]

    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6761))
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)

        start_time = time.time()
        # TODO: 适配流式输入
        if isinstance(text, Union[Generator, AsyncGenerator]):
            raise Exception(f'暂不支持流式输入')
        else:
            text = tensor_to_list(text + torch.tensor(6761))
            prompt_token_ids = [self.sos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            min_tokens = len(text) * 2
            max_tokens = len(text) * 20
            # async for output in self.llm_inference(
            #         prompt_token_ids,
            #         request_id=uuid,
            #         stop_token_ids=self.stop_token_ids,
            #         min_tokens=min_tokens,
            #         max_tokens=max_tokens,
            # ):
            #     if output.token_ids[-1] in self.stop_token_ids:
            #         need_add_tokens = output.token_ids[:-1]
            #     else:
            #         need_add_tokens = output.token_ids
            #     self.tts_speech_token_dict[uuid].extend(need_add_tokens)

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
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
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