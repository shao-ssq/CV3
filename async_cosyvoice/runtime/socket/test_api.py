# -*- coding: UTF-8 -*-
"""
CosyVoice Socket服务接口测试示例
包含所有HTTP和WebSocket接口的请求示例
"""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import aiohttp
import websockets


# 配置
BASE_URL = "http://localhost:38041"
WS_URL = "ws://localhost:38041"

# 认证配置
APP_ID = "18390413"
TOKEN = "qxmYaDtPRB"


def generate_signature(app_id: str, nonce: str, token: str) -> str:
    """生成认证签名

    Args:
        app_id: 应用ID
        nonce: 随机字符串（通常使用时间戳）
        token: 密钥

    Returns:
        signature: md5(md5(app_id + nonce) + token)
    """
    # 第一步：md5(app_id + nonce)
    md5_hash = hashlib.md5()
    input_string = app_id + nonce
    md5_hash.update(input_string.encode())
    digest = md5_hash.hexdigest()

    # 第二步：md5(digest + token)
    md5_hash = hashlib.md5()
    input_string = digest + token
    md5_hash.update(input_string.encode())
    signature = md5_hash.hexdigest()

    return signature


def get_auth_headers() -> dict:
    """获取认证请求头

    Returns:
        包含认证信息的headers字典
    """
    nonce = str(int(time.time() * 1000))  # 使用时间戳作为nonce
    signature = generate_signature(APP_ID, nonce, TOKEN)

    return {
        "app_id": APP_ID,
        "nonce": nonce,
        "signature": signature
    }


# ==================== HTTP接口示例 ====================

async def test_text_to_speech():
    """测试文本转语音接口"""
    print("\n=== 测试 POST /v1/audio/speech ===")

    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "input": "你好，这是一个测试语音。",
        "voice": "003",
        "response_format": "mp3",  # 可选: pcm, wav, mp3
        "speed": 1.0,
        "sample_rate": 24000,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                # 保存音频文件
                audio_data = await response.read()
                output_file = "output_speech.mp3"
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                print(f"✓ 音频已保存到: {output_file}, 大小: {len(audio_data)} bytes")
            else:
                print(f"✗ 请求失败: {response.status}")
                print(await response.text())


async def test_text_to_speech_stream():
    """测试流式文本转语音接口"""
    print("\n=== 测试 POST /v1/audio/speech_stream ===")

    url = f"{BASE_URL}/v1/audio/speech_stream"
    payload = {
        "input": ["你好", "这是", "流式", "输入", "测试"],
        "voice": "003",
        "response_format": "mp3",
        "speed": 1.0,
        "sample_rate": 24000,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                # 流式接收音频
                output_file = "output_stream.mp3"
                total_size = 0
                with open(output_file, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024):
                        f.write(chunk)
                        total_size += len(chunk)
                print(f"✓ 流式音频已保存到: {output_file}, 总大小: {total_size} bytes")
            else:
                print(f"✗ 请求失败: {response.status}")


async def test_voice_upload():
    """测试音色上传接口"""
    print("\n=== 测试 POST /v1/audio/voice/upload ===")

    url = f"{BASE_URL}/v1/audio/voice/upload"

    # 准备上传文件（需要一个实际的WAV文件）
    prompt_text = "希望你以后能够做的比我还好呦。。"
    voice_id = "女1"

    # 假设有一个测试音频文件
    test_audio_path = "/root/PycharmProjects/CV3/async_cosyvoice/wav/女1.wav"

    data = aiohttp.FormData()
    data.add_field('promptText', prompt_text)
    data.add_field('voiceId', voice_id)
    data.add_field('promptWav',
                   open(test_audio_path, 'rb'),
                   filename='prompt.wav',
                   content_type='audio/wav')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            result = await response.json()
            print(f"响应: {result}")
            if result.get('code') == '00000000':
                print(f"✓ 音色上传成功: {result.get('data')}")
            else:
                print(f"✗ 音色上传失败: {result.get('msg')}")


async def test_voice_list():
    """测试获取音色列表接口"""
    print("\n=== 测试 POST /v1/audio/voice/list ===")

    url = f"{BASE_URL}/v1/audio/voice/list"

    async with aiohttp.ClientSession() as session:
        async with session.post(url) as response:
            result = await response.json()
            print(f"响应: {result}")
            if result.get('code') == '00000000':
                voices = result.get('data', {}).get('voices', [])
                print(f"✓ 可用音色列表 ({len(voices)}个):")
                for voice in voices:
                    print(f"  - {voice}")
            else:
                print(f"✗ 获取失败: {result.get('msg')}")


async def test_record_upload():
    """测试录音上传接口"""
    print("\n=== 测试 POST /v1/record/upload ===")

    url = f"{BASE_URL}/v1/record/upload"

    voice_id = "user_001"
    voice_md5 = "test_md5_12345"
    test_audio_path = "/root/PycharmProjects/CV3/async_cosyvoice/runtime/async_grpc/zero_shot_prompt.wav"

    if not os.path.exists(test_audio_path):
        print(f"✗ 测试录音文件不存在: {test_audio_path}")
        print("  请准备一个WAV格式的录音文件用于测试")
        return

    data = aiohttp.FormData()
    data.add_field('voiceId', voice_id)
    data.add_field('voiceMd5', voice_md5)
    data.add_field('voiceFile',
                   open(test_audio_path, 'rb'),
                   filename='record.wav',
                   content_type='audio/wav')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            result = await response.json()
            print(f"响应: {result}")
            if result.get('code') == '00000000':
                print(f"✓ 录音上传成功")
            else:
                print(f"✗ 录音上传失败: {result.get('msg')}")


# ==================== WebSocket接口示例 ====================

async def test_websocket_voip_single(session_index: int, voiceId: str = "女1"):
    """单个VOIP WebSocket测试任务

    Args:
        session_index: 会话索引
        voiceId: 音色ID

    Returns:
        tuple: (是否成功, 耗时, 错误信息)
    """
    session_id = f"test_voip_session_{session_index:04d}"
    order_id = f"order_{session_index:04d}"

    # 连接音频接收端
    receiver_uri = f"{WS_URL}/ws/audio/{session_id}"
    # 连接音频请求端
    sender_uri = f"{WS_URL}/ws/{session_id}"

    # 获取认证headers
    auth_headers = get_auth_headers()

    start_time = time.time()

    try:
        # 同时建立两个连接（带认证headers）
        async with websockets.connect(receiver_uri, extra_headers=auth_headers) as receiver_ws, \
                   websockets.connect(sender_uri, extra_headers=auth_headers) as sender_ws:

            # 创建接收任务
            async def receive_audio():
                """接收音频数据"""
                audio_chunks = []
                first_chunk_time = None
                while True:
                    try:
                        message = await asyncio.wait_for(receiver_ws.recv(), timeout=10.0)
                        data = json.loads(message)
                        event = data.get('event')

                        if event == 'audio_data':
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                            audio_data = data.get('data', {}).get('audio', [])
                            audio_chunks.append(audio_data)
                        elif event == 'end':
                            return True, first_chunk_time
                        elif event == 'error':
                            error_data = data.get('data', {})
                            return False, None
                    except asyncio.TimeoutError:
                        return False, None
                    except Exception as e:
                        return False, None
                return False, None

            # 启动接收任务
            receive_task = asyncio.create_task(receive_audio())

            # 等待一小段时间确保接收端准备好
            await asyncio.sleep(0.1)

            # 发送请求
            start_message = {
                "event": "start",
                "orderId": order_id,
                "data": {
                    "voiceId": voiceId,
                    "inputStreaming": False,
                    "callType": "VOIP",
                    "voiceMd5": None
                }
            }
            await sender_ws.send(json.dumps(start_message))

            # 等待服务器处理 start 事件
            await asyncio.sleep(0.05)

            # 发送文本数据
            text_chunks = ["你好，", "这是", "VOIP", "模式", "测试。"]
            for text in text_chunks:
                text_message = {
                    "event": "text_data",
                    "orderId": order_id,
                    "data": {
                        "text": text
                    }
                }
                await sender_ws.send(json.dumps(text_message))
                await asyncio.sleep(0.01)

            # 等待所有文本数据发送完成
            await asyncio.sleep(0.05)

            # 发送结束事件
            end_message = {
                "event": "end",
                "orderId": order_id
            }
            await sender_ws.send(json.dumps(end_message))

            # 等待接收完成
            success, first_chunk_time = await receive_task

            end_time = time.time()
            total_time = end_time - start_time
            first_chunk_latency = first_chunk_time - start_time if first_chunk_time else None

            if success:
                return True, total_time, first_chunk_latency, None
            else:
                return False, total_time, None, "接收失败"

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        return False, total_time, None, str(e)


async def test_websocket_voip(concurrency: int = 10, total: int = 100, voiceId: str = "女1"):
    """测试VOIP WebSocket接口（支持并发测试）

    Args:
        concurrency: 并发数
        total: 总请求数
        voiceId: 音色ID
    """
    print("\n=== 测试 WebSocket VOIP模式（并发测试） ===")
    print(f"并发数: {concurrency}")
    print(f"总请求数: {total}")
    print(f"音色ID: {voiceId}")
    print("=" * 60)

    # 信号量控制并发
    semaphore = asyncio.Semaphore(concurrency)

    # 统计信息
    success_count = 0
    fail_count = 0
    total_times = []
    first_chunk_latencies = []
    errors = []

    async def worker(index: int):
        """工作协程"""
        nonlocal success_count, fail_count

        async with semaphore:
            success, total_time, first_chunk_latency, error = await test_websocket_voip_single(index, voiceId)

            if success:
                success_count += 1
                total_times.append(total_time)
                if first_chunk_latency:
                    first_chunk_latencies.append(first_chunk_latency)
                print(f"✓ [{index+1}/{total}] 成功 - 总耗时: {total_time:.3f}s, 首包延迟: {first_chunk_latency:.3f}s" if first_chunk_latency else f"✓ [{index+1}/{total}] 成功 - 总耗时: {total_time:.3f}s")
            else:
                fail_count += 1
                errors.append(error)
                print(f"✗ [{index+1}/{total}] 失败 - 耗时: {total_time:.3f}s, 错误: {error}")

    # 记录开始时间
    overall_start = time.time()

    # 创建所有任务
    tasks = [worker(i) for i in range(total)]

    # 执行所有任务
    await asyncio.gather(*tasks)

    # 记录结束时间
    overall_end = time.time()
    overall_time = overall_end - overall_start

    # 打印统计信息
    print("\n" + "=" * 60)
    print("测试统计")
    print("=" * 60)
    print(f"总请求数: {total}")
    print(f"成功数: {success_count}")
    print(f"失败数: {fail_count}")
    print(f"成功率: {success_count/total*100:.2f}%")
    print(f"总耗时: {overall_time:.3f}s")
    print(f"QPS: {total/overall_time:.2f}")

    if total_times:
        print(f"\n请求耗时统计:")
        print(f"  平均耗时: {sum(total_times)/len(total_times):.3f}s")
        print(f"  最小耗时: {min(total_times):.3f}s")
        print(f"  最大耗时: {max(total_times):.3f}s")

    if first_chunk_latencies:
        print(f"\n首包延迟统计:")
        print(f"  平均延迟: {sum(first_chunk_latencies)/len(first_chunk_latencies):.3f}s")
        print(f"  最小延迟: {min(first_chunk_latencies):.3f}s")
        print(f"  最大延迟: {max(first_chunk_latencies):.3f}s")

    if errors:
        print(f"\n错误统计:")
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            print(f"  {error}: {count}次")

    print("=" * 60)


async def test_websocket_outcall():
    """测试外呼WebSocket接口（音频请求 + 外呼接收）"""
    print("\n=== 测试 WebSocket 外呼模式 ===")

    session_id = "test_outcall_session_001"
    order_id = "order_outcall_001"

    # 连接外呼接收端
    receiver_uri = f"{WS_URL}/ws/ob/{session_id}"
    # 连接音频请求端
    sender_uri = f"{WS_URL}/ws/{session_id}"

    # 获取认证headers
    auth_headers = get_auth_headers()

    try:
        async with websockets.connect(receiver_uri, extra_headers=auth_headers) as receiver_ws, \
                   websockets.connect(sender_uri, extra_headers=auth_headers) as sender_ws:

            print(f"✓ WebSocket连接建立成功")

            # 创建接收任务
            async def receive_audio():
                """接收PCM音频数据（二进制）"""
                chunk_count = 0
                total_bytes = 0
                pcm_chunks = []
                while True:
                    try:
                        # 外呼模式接收二进制数据
                        message = await asyncio.wait_for(receiver_ws.recv(), timeout=10.0)

                        if isinstance(message, bytes):
                            # 二进制音频数据
                            chunk_count += 1
                            total_bytes += len(message)
                            pcm_chunks.append(message)
                            print(f"  [接收端] PCM音频块 #{chunk_count}, 大小: {len(message)} bytes")
                        else:
                            # JSON事件消息
                            data = json.loads(message)
                            event = data.get('event')
                            print(f"  [接收端] 事件: {event}")

                            if event == 'end':
                                print(f"  [接收端] 音频接收完成，共 {chunk_count} 个块，总大小: {total_bytes} bytes")
                                # 保存PCM音频文件
                                if pcm_chunks:
                                    all_pcm_bytes = b''.join(pcm_chunks)
                                    output_file = f"output_outcall_{session_id}.pcm"
                                    with open(output_file, "wb") as f:
                                        f.write(all_pcm_bytes)
                                    print(f"  [接收端] PCM音频已保存到: {output_file}, 大小: {len(all_pcm_bytes)} bytes")
                                    print(f"  [提示] PCM播放命令: ffplay -f s16le -ar 8000 -ac 1 {output_file}")
                                break
                            elif event == 'error':
                                error_data = data.get('data', {})
                                print(f"  [接收端] 错误: {error_data}")
                                break
                    except asyncio.TimeoutError:
                        print(f"  [接收端] 接收超时")
                        break
                    except Exception as e:
                        print(f"  [接收端] 异常: {e}")
                        break

            # 启动接收任务
            receive_task = asyncio.create_task(receive_audio())

            # 等待接收端准备好
            await asyncio.sleep(0.5)

            # 发送请求
            print(f"  [发送端] 发送start事件")
            start_message = {
                "event": "start",
                "orderId": order_id,
                "data": {
                    "voiceId": "001",  # 使用可用的音色
                    "inputStreaming": False,
                    "callType": "OUT_CALL",  # 外呼模式
                    "voiceMd5": None
                }
            }
            await sender_ws.send(json.dumps(start_message))

            # 发送文本数据
            print(f"  [发送端] 发送文本数据")
            text_message = {
                "event": "text_data",
                "orderId": order_id,
                "data": {
                    "text": "您好，这是外呼模式测试，音频将以8000Hz的PCM格式传输。"
                }
            }
            await sender_ws.send(json.dumps(text_message))

            # 发送结束事件
            print(f"  [发送端] 发送end事件")
            end_message = {
                "event": "end",
                "orderId": order_id
            }
            await sender_ws.send(json.dumps(end_message))

            # 等待接收完成
            await receive_task

            print(f"✓ 外呼模式测试完成")

    except Exception as e:
        print(f"✗ WebSocket连接失败: {e}")


async def test_websocket_streaming():
    """测试流式输入WebSocket接口"""
    print("\n=== 测试 WebSocket 流式输入模式 ===")

    session_id = "test_streaming_session_001"
    order_id = "order_streaming_001"

    receiver_uri = f"{WS_URL}/ws/audio/{session_id}"
    sender_uri = f"{WS_URL}/ws/{session_id}"

    # 获取认证headers
    auth_headers = get_auth_headers()

    try:
        async with websockets.connect(receiver_uri, extra_headers=auth_headers) as receiver_ws, \
                   websockets.connect(sender_uri, extra_headers=auth_headers) as sender_ws:

            print(f"✓ WebSocket连接建立成功")

            # 创建接收任务
            async def receive_audio():
                audio_chunks = []
                while True:
                    try:
                        message = await asyncio.wait_for(receiver_ws.recv(), timeout=15.0)
                        data = json.loads(message)
                        event = data.get('event')

                        if event == 'audio_data':
                            audio_data = data.get('data', {}).get('audio', [])
                            seq_no = data.get('data', {}).get('seqNo', 0)
                            audio_chunks.append(audio_data)
                            print(f"  [接收端] 音频块 #{seq_no}")
                        elif event == 'end':
                            print(f"  [接收端] 完成，共 {len(audio_chunks)} 个块")
                            # 保存音频文件
                            if audio_chunks:
                                all_audio_bytes = b''.join([bytes(chunk) for chunk in audio_chunks])
                                output_file = f"output_streaming_{session_id}.mp3"
                                with open(output_file, "wb") as f:
                                    f.write(all_audio_bytes)
                                print(f"  [接收端] 音频已保存到: {output_file}, 大小: {len(all_audio_bytes)} bytes")
                            break
                        elif event == 'error':
                            print(f"  [接收端] 错误: {data.get('data')}")
                            break
                    except asyncio.TimeoutError:
                        break

            receive_task = asyncio.create_task(receive_audio())
            await asyncio.sleep(0.5)

            # 发送start事件，开启流式输入
            print(f"  [发送端] 发送start事件（流式输入）")
            start_message = {
                "event": "start",
                "orderId": order_id,
                "data": {
                    "voiceId": "001",  # 使用可用的音色
                    "inputStreaming": True,  # 流式输入
                    "callType": "VOIP",
                    "voiceMd5": None
                }
            }
            await sender_ws.send(json.dumps(start_message))

            # 逐步发送文本（模拟实时输入）
            text_parts = ["今天", "天气", "很好，", "适合", "出门", "散步。"]
            for i, text in enumerate(text_parts):
                print(f"  [发送端] 流式发送文本 #{i}: {text}")
                text_message = {
                    "event": "text_data",
                    "orderId": order_id,
                    "data": {"text": text}
                }
                await sender_ws.send(json.dumps(text_message))
                # 模拟实时输入延迟
                await asyncio.sleep(0.3)

            # 发送结束
            print(f"  [发送端] 发送end事件")
            end_message = {
                "event": "end",
                "orderId": order_id
            }
            await sender_ws.send(json.dumps(end_message))

            await receive_task
            print(f"✓ 流式输入测试完成")

    except Exception as e:
        print(f"✗ WebSocket连接失败: {e}")


async def test_websocket_cancel():
    """测试取消WebSocket请求"""
    print("\n=== 测试 WebSocket 取消功能 ===")

    session_id = "test_cancel_session_001"
    order_id = "order_cancel_001"

    receiver_uri = f"{WS_URL}/ws/audio/{session_id}"
    sender_uri = f"{WS_URL}/ws/{session_id}"

    # 获取认证headers
    auth_headers = get_auth_headers()

    try:
        async with websockets.connect(receiver_uri, extra_headers=auth_headers) as receiver_ws, \
                   websockets.connect(sender_uri, extra_headers=auth_headers) as sender_ws:

            print(f"✓ WebSocket连接建立成功")

            # 发送start
            start_message = {
                "event": "start",
                "orderId": order_id,
                "data": {
                    "voiceId": "001",  # 使用可用的音色
                    "inputStreaming": False,
                    "callType": "VOIP"
                }
            }
            await sender_ws.send(json.dumps(start_message))
            print(f"  [发送端] 发送start事件")

            # 发送一些文本
            text_message = {
                "event": "text_data",
                "orderId": order_id,
                "data": {"text": "这是一个很长的文本，但是我们会在中途取消它..."}
            }
            await sender_ws.send(json.dumps(text_message))
            print(f"  [发送端] 发送文本数据")

            # 短暂等待后取消
            await asyncio.sleep(0.2)

            cancel_message = {
                "event": "cancel",
                "orderId": order_id
            }
            await sender_ws.send(json.dumps(cancel_message))
            print(f"  [发送端] 发送cancel事件")

            print(f"✓ 取消功能测试完成")

    except Exception as e:
        print(f"✗ WebSocket连接失败: {e}")


# ==================== 主函数 ====================

async def main():
    """运行所有测试"""
    print("=" * 60)
    print("CosyVoice Socket服务接口测试")
    print("=" * 60)

    # 测试HTTP接口
    # print("\n【HTTP接口测试】")
    # await test_text_to_speech()
    # await test_text_to_speech_stream()
    await test_voice_upload()  # 需要准备测试音频文件
    await test_voice_list()
    # await test_record_upload()  # 需要准备测试音频文件

    # 测试WebSocket接口
    print("\n【WebSocket接口测试】")
    # 并发测试：并发数=10，总请求数=100
    await test_websocket_voip(concurrency=4, total=100, voiceId="女1")
    # await test_websocket_outcall()
    # await test_websocket_streaming()
    # await test_websocket_cancel()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
