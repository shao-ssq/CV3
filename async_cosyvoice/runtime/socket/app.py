# -*- coding: UTF-8 -*-
"""
CosyVoice WebSocket服务主应用
"""
import argparse
import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.types import ASGIApp, Scope, Receive, Send

from async_cosyvoice.config import logger, record_upload_dir, record_hash_count

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice3

# 导入模块
from models import SpeechRequest, HttpResponse, SpeakerAddException, StreamSpeechRequest
from utils import (
    convert_audio_tensor_to_bytes, generator_wrapper, get_content_type,
    load_wav, auth_validate, load_wav_and_upsample, hash_index
)
from websocket_handler import ConnectionManager, AudioConnectionManager


# 全局变量
cosyvoice = None
receiver = None
ob_receiver = None
manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动")
    # 如果cosyvoice有warmup和setup方法，调用它们
    if hasattr(cosyvoice, 'warmup'):
        await cosyvoice.warmup()
    if hasattr(cosyvoice, 'setup_timer_task'):
        await cosyvoice.setup_timer_task()
    yield
    # 应用关闭
    if hasattr(cosyvoice, 'shutdown'):
        await cosyvoice.shutdown()
    logger.info("应用关闭")


app = FastAPI(lifespan=lifespan)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=16)


class WebSocketAuthMiddleware:
    """WebSocket认证中间件"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            # 获取 headers
            headers = dict(scope.get("headers", []))
            app_id = headers.get(b"app_id")
            nonce = headers.get(b"nonce")
            signature = headers.get(b"signature")

            if not auth_validate(app_id, nonce, signature):
                # 拒绝连接
                await send({
                    "type": "websocket.close",
                    "code": 1008  # Policy Violation
                })
                return

        await self.app(scope, receive, send)


app.add_middleware(WebSocketAuthMiddleware)


async def generate_audio_content(request: SpeechRequest) -> AsyncGenerator[bytes, Any]:
    """生成音频内容（异步生成器）"""
    tts_text = request.input
    spk_id = request.voice

    try:
        end_of_prompt_index = tts_text.find("<|endofprompt|>")
        if end_of_prompt_index != -1:
            instruct_text = tts_text[: end_of_prompt_index]
            tts_text = tts_text[end_of_prompt_index + len("<|endofprompt|>"):]

            audio_tensor_data_generator = generator_wrapper(cosyvoice.inference_instruct2_by_spk_id(
                tts_text,
                instruct_text,
                spk_id,
                stream=request.stream,
                speed=request.speed,
                text_frontend=True
            ), request.sample_rate)
        else:
            audio_tensor_data_generator = generator_wrapper(cosyvoice.inference_zero_shot_by_spk_id(
                tts_text,
                spk_id,
                stream=request.stream,
                speed=request.speed,
                text_frontend=True
            ), request.sample_rate)

        audio_bytes_data_generator = convert_audio_tensor_to_bytes(
            audio_tensor_data_generator,
            request.response_format,
            sample_rate=request.sample_rate,
            stream=request.stream,
        )

        # 使用 async for 来 yield 音频数据块
        async for chunk in audio_bytes_data_generator:
            yield chunk

    except Exception as e:
        logger.error(f"生成音频失败: {e}")
        raise e


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    """文本转语音接口"""
    try:
        # 构建响应头
        content_type = get_content_type(
            request.response_format,
            request.sample_rate
        )
        voice_id = str(uuid.uuid4())
        filename = f"{voice_id}.{request.response_format}"

        # 返回流式响应（不要await生成器）
        return StreamingResponse(
            content=generate_audio_content(request),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"TTS接口错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/v1/record/upload")
async def speech_upload(voiceId: str = Form(), voiceMd5: str = Form(), voiceFile: UploadFile = File()):
    """录音上传接口"""
    try:
        loop = asyncio.get_event_loop()
        speech = await loop.run_in_executor(
            executor,
            load_wav_and_upsample,
            voiceFile.file,
            24000
        )

        record_dir = os.path.join(record_upload_dir, voiceId, str(hash_index(voiceMd5, record_hash_count)))
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir, exist_ok=True)

        torch.save({"tts_speech": speech.cpu()}, os.path.join(record_dir, voiceMd5 + '.pt'))
        logger.info(f"[{voiceId}][{voiceMd5}]录音上传完成")
        return HttpResponse.success(None)
    except Exception as e:
        logger.error(f"[{voiceId}][{voiceMd5}]录音上传失败:{e}")
        return HttpResponse.fail("00000001", f"录音上传失败:{e}")


@app.post("/v1/audio/speech_stream")
async def text_to_speech_stream(request: StreamSpeechRequest):
    """流式文本转语音接口"""
    try:
        # 构建响应头
        content_type = get_content_type(
            request.response_format,
            request.sample_rate
        )
        voice_id = str(uuid.uuid4())
        filename = f"{voice_id}.{request.response_format}"

        async def tts_text(arr):
            for item in arr:
                yield item

        audio_tensor_data_generator = generator_wrapper(cosyvoice.inference_zero_shot_by_spk_id(
            tts_text(request.input),
            request.voice,
            stream=request.stream,
            speed=request.speed,
            text_frontend=True
        ), request.sample_rate)

        audio_bytes_data_generator = convert_audio_tensor_to_bytes(
            audio_tensor_data_generator,
            request.response_format,
            sample_rate=request.sample_rate,
            stream=request.stream,
        )

        # 返回流式响应
        return StreamingResponse(
            content=audio_bytes_data_generator,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"流式TTS接口错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/v1/audio/voice/upload")
async def add_speaker(promptText: str = Form(), promptWav: UploadFile = File(), voiceId: str = Form()):
    """添加音色接口"""
    try:
        logger.info(f"上传音色{voiceId}")
        prompt_speech_16k = load_wav(promptWav.file, 16000)
        # 使用frontend生成音色信息
        cosyvoice.frontend.generate_spk_info(voiceId, promptText, prompt_speech_16k, cosyvoice.sample_rate, "A")
        logger.info(f"上传音色{voiceId}成功")
        return HttpResponse.success({"voiceId": voiceId})
    except Exception as e:
        logger.info(f"上传音色{voiceId}失败:{e}")
        return HttpResponse.fail("00000001", str(e))


@app.post("/v1/audio/voice/list")
# 懒加载, 保存在磁盘上的音色文件（如 spk2info/002.pt、spk2info/003.pt）不会自动加载到内存
async def voice_list():
    """获取音色列表"""
    try:
        spks = cosyvoice.list_available_spks()
        return HttpResponse.success({"voices": spks})
    except Exception as e:
        logger.error(f"获取音色列表失败:{e}")
        return HttpResponse.fail("00000001", str(e))


@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """音频接收端WebSocket"""
    await receiver.connect(websocket, session_id)
    try:
        while True:
            message = await websocket.receive_text()
            await receiver.handle_message(session_id, message)
    except WebSocketDisconnect as e:
        receiver.disconnect(session_id, e)
    except Exception as e:
        logger.error(e)


@app.websocket("/ws/ob/{session_id}")
async def websocket_ob_endpoint(websocket: WebSocket, session_id: str):
    """外呼接收端WebSocket"""
    await ob_receiver.connect(websocket, session_id)
    try:
        while True:
            message = await websocket.receive_text()
            await ob_receiver.handle_message(session_id, message)
    except WebSocketDisconnect as e:
        ob_receiver.disconnect(session_id, e)
    except Exception as e:
        logger.error(e)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """音频请求端WebSocket"""
    await manager.connect(websocket, session_id)

    try:
        while True:
            message = await websocket.receive_text()
            await manager.handle_message(session_id, message)
    except WebSocketDisconnect as e:
        manager.disconnect(session_id, e)
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=38041)
    parser.add_argument('--model_dir',
                        type=str,
                        default='/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--load_trt',
                        type=bool,
                        default=True,
                        help='whether to load tensorrt model')
    parser.add_argument('--fp16',
                        type=bool,
                        default=True,
                        help='whether to use fp16')
    args = parser.parse_args()

    # 初始化CosyVoice
    cosyvoice = AsyncCosyVoice3(args.model_dir, load_trt=args.load_trt, fp16=args.fp16)
    receiver = ConnectionManager("音频接收端")
    ob_receiver = ConnectionManager("外呼接收端")
    manager = AudioConnectionManager("音频请求端", receiver, ob_receiver, cosyvoice)

    uvicorn.run(app, host="0.0.0.0", port=args.port, ws_max_size=67108864, ws_max_queue=128,
                timeout_graceful_shutdown=30, h11_max_incomplete_event_size=32 * 1024 * 1024,
                loop="uvloop")
