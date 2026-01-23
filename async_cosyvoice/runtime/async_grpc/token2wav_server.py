"""
Token2Wav gRPC 服务端
独立部署，用于分布式 token → wav 转换
"""
import os
import sys
import signal
import asyncio
import argparse
import logging
from typing import AsyncIterator
from concurrent import futures

import torch
import numpy as np
import grpc
from grpc import aio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')

import cosyvoice_pb2
import cosyvoice_pb2_grpc
from async_cosyvoice.token2wav_model import Token2WavModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def deserialize_tensor(data: bytes, meta) -> torch.Tensor:
    """反序列化 tensor"""
    if not data or not meta.shape:
        return torch.zeros(0)
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int32': np.int32,
        'int64': np.int64,
    }
    dtype = dtype_map.get(meta.dtype, np.float32)
    shape = tuple(meta.shape)
    arr = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


def serialize_audio(audio: torch.Tensor, format: str = 'raw_float32') -> bytes:
    """序列化音频数据"""
    if format == 'pcm_int16':
        audio_np = (audio.cpu().numpy() * 32767).astype(np.int16)
    else:
        audio_np = audio.cpu().numpy().astype(np.float32)
    return audio_np.tobytes()


class Token2WavServiceImpl(cosyvoice_pb2_grpc.Token2WavServiceServicer):
    """Token2Wav gRPC 服务实现"""

    def __init__(self, args):
        self.model = Token2WavModel(
            model_dir=args.model_dir,
            load_jit=args.load_jit,
            load_trt=args.load_trt,
            fp16=args.fp16,
            device=args.device,
            estimator_count=args.estimator_count
        )
        self.executor = futures.ThreadPoolExecutor(max_workers=args.max_workers)
        logging.info(f'Token2Wav service ready, max_workers={args.max_workers}')

    async def Convert(
        self,
        request: cosyvoice_pb2.Token2WavRequest,
        context: aio.ServicerContext
    ) -> cosyvoice_pb2.Token2WavResponse:
        """单次转换"""
        try:
            speech_tokens = torch.tensor(list(request.speech_tokens)).unsqueeze(0)
            prompt_token = deserialize_tensor(request.prompt_token, request.prompt_token_meta)
            prompt_feat = deserialize_tensor(request.prompt_feat, request.prompt_feat_meta)
            embedding = deserialize_tensor(request.embedding, request.embedding_meta)

            session_id = request.session_id or 'default'
            loop = asyncio.get_event_loop()

            audio = await loop.run_in_executor(
                self.executor,
                self.model.token2wav,
                speech_tokens,
                prompt_token,
                prompt_feat,
                embedding,
                request.token_offset,
                session_id,
                request.stream,
                request.finalize,
                request.speed or 1.0
            )

            if not request.stream or request.finalize:
                self.model.clear_session(session_id)

            audio_bytes = serialize_audio(audio, 'raw_float32')
            return cosyvoice_pb2.Token2WavResponse(
                audio_data=audio_bytes,
                format='raw_float32',
                sample_rate=self.model.sample_rate
            )

        except Exception as e:
            logging.error(f'Convert failed: {e}', exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def StreamConvert(
        self,
        request_iterator: AsyncIterator[cosyvoice_pb2.Token2WavRequest],
        context: aio.ServicerContext
    ) -> AsyncIterator[cosyvoice_pb2.Token2WavResponse]:
        """流式转换"""
        session_id = None
        try:
            async for request in request_iterator:
                session_id = request.session_id

                speech_tokens = torch.tensor(list(request.speech_tokens)).unsqueeze(0)
                prompt_token = deserialize_tensor(request.prompt_token, request.prompt_token_meta)
                prompt_feat = deserialize_tensor(request.prompt_feat, request.prompt_feat_meta)
                embedding = deserialize_tensor(request.embedding, request.embedding_meta)

                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(
                    self.executor,
                    self.model.token2wav,
                    speech_tokens,
                    prompt_token,
                    prompt_feat,
                    embedding,
                    request.token_offset,
                    session_id,
                    True,
                    request.finalize,
                    request.speed or 1.0
                )

                audio_bytes = serialize_audio(audio, 'raw_float32')
                yield cosyvoice_pb2.Token2WavResponse(
                    audio_data=audio_bytes,
                    format='raw_float32',
                    sample_rate=self.model.sample_rate
                )

        except Exception as e:
            logging.error(f'StreamConvert failed: {e}', exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            if session_id:
                self.model.clear_session(session_id)


async def serve(args):
    """启动服务"""
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]

    server = aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=options,
        maximum_concurrent_rpcs=args.max_conc
    )

    cosyvoice_pb2_grpc.add_Token2WavServiceServicer_to_server(
        Token2WavServiceImpl(args), server
    )

    server.add_insecure_port(f'0.0.0.0:{args.port}')
    await server.start()
    logging.info(f'Token2Wav server listening on 0.0.0.0:{args.port}')

    # 信号处理
    async def shutdown(sig):
        logging.info(f'Received signal {sig.name}, shutting down...')
        await server.stop(5)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s))
        )

    await server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description='Token2Wav gRPC Server')
    parser.add_argument('--port', type=int, default=50001)
    parser.add_argument('--model_dir', type=str,
                        default='/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--device', type=str, default=None,
                        help='CUDA device, e.g., cuda:0')
    parser.add_argument('--load_jit', action='store_true')
    parser.add_argument('--load_trt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--max_conc', type=int, default=50)
    parser.add_argument('--estimator_count', type=int, default=4)

    args = parser.parse_args()

    try:
        asyncio.run(serve(args))
    except asyncio.CancelledError:
        logging.info('Server shutdown complete.')


if __name__ == '__main__':
    main()
