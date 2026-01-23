# -*- coding: UTF-8 -*-
"""
WebSocket连接管理和处理逻辑
"""
import asyncio
import json
import os
import time
from typing import AsyncGenerator, Dict

import aiofiles
import numpy as np
from fastapi import WebSocket

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice3
from async_cosyvoice.config import logger, save_sentence_audio, sentence_audio_dir
from models import (
    SpeakerNotExistException, ReceiverNotConnectedException, TTSException,
    AudioError, AudioEvent, CallType, AudioFormat
)
from utils import convert_audio_tensor_to_bytes, generator_wrapper


class OrderedTextBuffer:
    """用于按顺序缓存和产出文本的类"""

    def __init__(self, session_id: str, order_id: str):
        self.order_id = order_id
        self.session_id = session_id
        self.buffer = asyncio.Queue(maxsize=1000)  # 使用异步队列存储文本
        self.is_completed = False  # 是否收到结束信号
        self.full_text = ""

    async def add_text(self, text: str):
        """添加文本到缓冲区"""
        logger.info(f"接收到[{self.session_id}]的合成文本{self.order_id}:{text}")
        try:
            await asyncio.wait_for(self.buffer.put(text), timeout=1.0)
        except asyncio.TimeoutError:
            logger.error(f"[{self.session_id}][{self.order_id}]缓冲区超限，无法添加文本")

    def mark_completed(self):
        """标记文本发送完成"""
        self.is_completed = True

    async def text_generator(self) -> AsyncGenerator[str, None]:
        """按顺序产出文本的异步生成器"""
        start_time = asyncio.get_event_loop().time()
        max_wait_time = 10.0  # 最大等待时间10秒

        while True:
            # 检查是否超时（客户端异常没有end事件）
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > max_wait_time:
                logger.warning(f"[{self.session_id}][{self.order_id}]文本生成器超时中断")
                break

            # 如果队列为空但已完成，则退出
            if self.is_completed and self.buffer.empty():
                break

            # 优先检查队列是否有数据，避免不必要的等待
            if not self.buffer.empty():
                text = await self.buffer.get()
                yield text
                self.full_text += text
                self.buffer.task_done()
                # 重置超时计时器，因为有数据流动
                start_time = current_time
                continue

            # 队列为空时，使用动态超时策略
            timeout = 0.003  # 默认3毫秒
            if self.is_completed:
                # 如果已标记完成，使用较短超时快速退出
                timeout = 0.001  # 1毫秒

            try:
                text = await asyncio.wait_for(self.buffer.get(), timeout=timeout)
                yield text
                self.full_text += text
                self.buffer.task_done()
                # 重置超时计时器，因为有数据流动
                start_time = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                # 超时后检查是否已完成
                if self.is_completed and self.buffer.empty():
                    break


class ConnectionManager:
    """基础WebSocket连接管理器"""

    def __init__(self, name: str):
        self.name = name
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        if session_id in self.active_connections:
            await websocket.send_json(
                {"event": AudioEvent.error.name, "caseId": session_id,
                 "data": AudioError.CONNECTION_EXISTS.to_json()})
            await websocket.close()
            return
        self.active_connections[session_id] = websocket
        logger.info(f"[{self.name}][{session_id}]建立连接")

    def disconnect(self, session_id: str, e):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        logger.info(f"[{self.name}][{session_id}]连接断开 {e}")

    async def send_message(self, session_id: str, message: dict):
        """发送JSON消息"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                logger.info(f"[{self.name}][{session_id}]发送消息 {message}")
                return
            except Exception as e:
                logger.error(f"[{self.name}][{session_id}]发送消息失败 {e}")
                raise ReceiverNotConnectedException(session_id)
        raise ReceiverNotConnectedException(session_id)

    async def send_event(self, session_id: str, order_id: str, event: AudioEvent, data: dict | None):
        """发送事件消息"""
        await self.send_message(session_id,
                                {"event": event.name, "caseId": session_id, "orderId": order_id, "data": data})

    async def send_error(self, session_id: str, order_id: str, error: AudioError, e: Exception):
        """发送错误消息"""
        try:
            logger.error(f"[{self.name}][{session_id}]{error.desc}")
            await self.send_event(session_id, order_id, AudioEvent.error,
                                  {"code": error.code, "message": f"{error.desc}:{e}"})
        except Exception as e:
            logger.error(f"[{self.name}][{session_id}]发送消息失败 {e}")

    async def send_bytes(self, session_id: str, order_id: str, index: int, audio_bytes: bytes):
        """发送二进制音频数据"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_bytes(audio_bytes)
                logger.info(f"[{self.name}][{session_id}][{order_id}]音频chunk{index}已发送")
                return
            except Exception as e:
                logger.error(f"[{self.name}][{session_id}]发送消息失败 {e}")
                raise ReceiverNotConnectedException(session_id)
        raise ReceiverNotConnectedException(session_id)

    async def handle_message(self, session_id: str, message: str):
        """处理收到的消息（基类默认实现）"""
        logger.info(f"[{self.name}][{session_id}]接收消息 {message}")


class AudioConnectionManager(ConnectionManager):
    """音频连接管理器，处理TTS请求"""

    def __init__(self, name: str, receiver: ConnectionManager, ob_receiver: ConnectionManager,
                 cosyvoice: AsyncCosyVoice3):
        super().__init__(name)
        self.text_buffers: Dict[str, OrderedTextBuffer] = {}  # order_id -> buffer
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.receiver = receiver
        self.ob_receiver = ob_receiver
        self.cosyvoice = cosyvoice

    def create_text_buffer(self, session_id: str, order_id: str):
        """创建文本缓冲区"""
        if order_id not in self.text_buffers:
            self.text_buffers[order_id] = OrderedTextBuffer(session_id, order_id)

    def get_text_buffer(self, session_id: str, order_id: str) -> OrderedTextBuffer:
        """获取文本缓冲区"""
        return self.text_buffers.get(order_id, None)

    async def handle_message(self, session_id: str, message: str):
        """处理音频请求消息"""
        await super().handle_message(session_id, message)
        try:
            message = json.loads(message)
            event = message.get('event')
            order_id = message.get('orderId')

            if not order_id:
                logger.info(f"[{self.name}][{session_id}]缺少orderId")
                return

            if event == AudioEvent.start.name:
                # 开始处理新的订单
                voice_id = message.get('data', {}).get('voiceId', 'default')
                input_streaming = message.get('data', {}).get('inputStreaming', False)
                call_type = message.get('data', {}).get('callType', 'OUT_CALL')
                voice_md5 = message.get('data', {}).get('voiceMd5', None)
                logger.info(f"[{self.name}][{session_id}]开始处理文本事件")
                # 如果已有处理任务，先取消
                if session_id in self.processing_tasks:
                    logger.info(f"[{self.name}][{session_id}][{order_id}]打断上一句请求")
                    self.processing_tasks[session_id].cancel()
                self.create_text_buffer(session_id, order_id)
                # 创建新的处理任务
                self.processing_tasks[session_id] = asyncio.create_task(
                    self.process_texts(session_id, order_id, voice_id, voice_md5, input_streaming, call_type))
            elif event == AudioEvent.cancel.name:
                if session_id in self.processing_tasks:
                    logger.info(f"[{self.name}][{session_id}][{order_id}]接收主动打断请求")
                    self.processing_tasks[session_id].cancel()
                    await asyncio.sleep(0.1)  # 给任务一点时间取消
            elif event == AudioEvent.text_data.name:
                # 接收文本数据
                text = message.get('data', {}).get('text')

                if text is None:
                    logger.warning(f"[{self.name}][{session_id}][{order_id}]text为空")
                    return

                # 将文本添加到有序缓冲区
                buffer = self.get_text_buffer(session_id, order_id)
                if buffer is not None:
                    await buffer.add_text(text)
                    logger.info(f"[{self.name}][{session_id}][{order_id}]添加文本[{text}]")
                else:
                    logger.info(f"[{self.name}][{session_id}][{order_id}]buffer不存在")

            elif event == AudioEvent.end.name:
                logger.info(f"[{self.name}][{session_id}][{order_id}]文本事件结束")
                # 文本发送完成
                if order_id in self.text_buffers:
                    buffer = self.text_buffers[order_id]
                    buffer.mark_completed()

            else:
                logger.warning(f"[{self.name}][{session_id}][{order_id}]无法识别的事件:{event}")

            logger.info(f"[{self.name}][{session_id}][{order_id}]{event}事件处理完成")

        except json.JSONDecodeError:
            logger.error(f"[{self.name}][{session_id}]json解析异常")
        except Exception as e:
            logger.error(f"[{self.name}][{session_id}]消息处理失败 {e}")

    async def process_texts(self, session_id: str, order_id: str, voice_id: str, voice_md5: str,
                            input_streaming: bool = False,
                            call_type: str = CallType.OUT_CALL.name):
        """处理文本缓冲区，生成音频"""
        audio_format = AudioFormat.mp3.name if call_type == CallType.VOIP.name else AudioFormat.pcm.name
        sample_rate = 24000 if call_type == CallType.VOIP.name else 8000
        receiver = self.receiver if call_type == CallType.VOIP.name else self.ob_receiver

        try:
            await receiver.send_event(session_id, order_id, AudioEvent.start, None)
            buffer = self.get_text_buffer(session_id, order_id)

            # 创建文件
            f = None
            if save_sentence_audio:
                f = await aiofiles.open(os.path.join(sentence_audio_dir, f"audio-{order_id}.{audio_format}"), "wb")

            # 对于非流式输入，等待所有文本数据到达
            if not input_streaming:
                # 等待 end 事件标记完成
                max_wait_time = 5.0  # 最多等待5秒
                start_wait = asyncio.get_event_loop().time()
                while not buffer.is_completed:
                    await asyncio.sleep(0.01)  # 每10毫秒检查一次
                    if asyncio.get_event_loop().time() - start_wait > max_wait_time:
                        logger.warning(f"[{session_id}][{order_id}]等待文本数据超时")
                        break

            # 创建有序的文本生成器
            input_text = ""
            if not input_streaming:
                async for text_chunk in buffer.text_generator():
                    input_text += text_chunk
            else:
                input_text = buffer.text_generator()

            i = 0
            start_time = time.time()
            audio_generator = generator_wrapper(
                self.cosyvoice.inference_zero_shot_by_spk_id(
                    input_text, voice_id,
                    stream=True,
                    speed=1,
                    text_frontend=True
                ),
                sample_rate
            )
            async for audio_bytes in convert_audio_tensor_to_bytes(audio_generator, audio_format,
                                                                   sample_rate=sample_rate, stream=True):

                if call_type == CallType.VOIP.name:
                    np_array = np.frombuffer(audio_bytes, dtype=np.uint8)
                    await receiver.send_event(session_id, order_id, AudioEvent.audio_data, {
                        "audio": np_array.tolist(),
                        "seqNo": i,
                        "text": buffer.full_text
                    })
                else:
                    if i == 0:
                        first_chunk_latency = int((time.time() - start_time) * 1000)
                        logger.info(
                            f"[{self.name}][{session_id}][{order_id}][{voice_id}]首包延迟{first_chunk_latency}ms")
                    await receiver.send_bytes(session_id, order_id, i, audio_bytes)
                if save_sentence_audio:
                    await f.write(audio_bytes)
                i += 1
            if save_sentence_audio:
                await f.close()

        except asyncio.CancelledError:
            logger.info(f"[{self.name}][{session_id}][{order_id}]任务被取消")

        except SpeakerNotExistException as e:
            await self.send_error(session_id, order_id, AudioError.SPK_NOT_EXISTS, e)

        except ReceiverNotConnectedException as e:
            await self.send_error(session_id, order_id, AudioError.RECEIVER_NOT_CONNECTED, e)

        except TTSException as e:
            await self.send_error(session_id, order_id, AudioError.TTS_ERROR, e)

        except Exception as e:
            logger.error(f"Error processing texts for order {order_id}: {e}")

        finally:
            logger.info(f"[{self.name}][{session_id}][{order_id}]清理临时资源")
            await receiver.send_event(session_id, order_id, AudioEvent.end, None)

            # 清理资源
            if order_id in self.text_buffers:
                del self.text_buffers[order_id]
            if session_id in self.processing_tasks:
                del self.processing_tasks[session_id]
