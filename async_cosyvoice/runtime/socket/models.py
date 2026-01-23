# -*- coding: UTF-8 -*-
"""
WebSocket服务的数据模型定义
"""
from enum import Enum, unique
from typing import Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated


# ============ 枚举类 ============
@unique
class AudioEvent(Enum):

    def __init__(self, desc: str):
        self.desc = desc

    start = '开始'
    cancel = '取消'
    end = '结束'
    error = '异常'
    text_data = '文本数据'
    audio_data = '音频数据'


@unique
class CallType(Enum):
    VOIP = "VOIP"
    OUT_CALL = "OUT_CALL"


@unique
class AudioFormat(Enum):
    wav = "wav"
    pcm = "pcm"
    mp3 = "mp3"


# ============ 错误定义 ============
@unique
class AudioError(Enum):
    def __init__(self, code: str, desc: str):
        self.code = code
        self.desc = desc

    RECEIVER_NOT_CONNECTED = ("0001", "音频接收端未连接")
    CONNECTION_EXISTS = ("0002", "连接已存在")
    SPK_NOT_EXISTS = ("0003", "音色不存在")
    TTS_ERROR = ("0004", "合成失败")

    def to_json(self):
        return {
            "code": self.code,
            "message": self.desc
        }


# ============ 自定义异常 ============
class SpeakerAddException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class SpeakerNotExistException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class TTSException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class ReceiverNotConnectedException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


# ============ 请求/响应模型 ============
class HttpResponse(BaseModel):
    code: str
    msg: str
    data: object

    @staticmethod
    def success(data=None):
        return HttpResponse(code="00000000", msg="操作成功", data=data)

    @staticmethod
    def fail(code: str, msg: str):
        return HttpResponse(code=code, msg=msg, data=None)


class SpeechRequest(BaseModel):
    """语音合成请求参数"""
    input: str = Field(
        ...,
        max_length=4096,
        examples=["你好，欢迎使用语音合成服务！"],
        description="需要转换为语音的文本内容"
    )
    voice: str = Field(
        ...,
        examples=[
            "001",
            "speech:voice-name:xxx:xxx",
        ],
        description="音色选择"
    )
    response_format: Optional[Literal["mp3", "wav", "pcm"]] = Field(
        "mp3",
        examples=["mp3", "wav", "pcm"],
        description="输出音频格式"
    )
    sample_rate: Optional[int] = Field(
        24000,
        description="采样率，目前不支持设置，默认为返回 24000 Hz音频数据"
    )
    stream: Optional[bool] = Field(
        False,
        description="开启流式返回"
    )
    speed: Annotated[Optional[float], Field(strict=True, ge=0.25, le=4.0)] = Field(
        1.0,
        description="语速控制[0.25-4.0]"
    )
    use_cache: Optional[bool] = Field(
        False,
        description="是否使用缓存"
    )
    voice_md5: Optional[str] = Field(
        None,
        description="录音md5"
    )


class StreamSpeechRequest(BaseModel):
    input: list[str] = Field(
        ...,
        description="文本数组"
    )
    voice: str = Field(
        ...,
        examples=[
            "001",
            "speech:voice-name:xxx:xxx",
        ],
        description="音色选择"
    )
    response_format: Optional[Literal["mp3", "wav", "pcm"]] = Field(
        "mp3",
        examples=["mp3", "wav", "pcm"],
        description="输出音频格式"
    )
    sample_rate: Optional[int] = Field(
        24000,
        description="采样率，目前不支持设置，默认为返回 24000 Hz音频数据"
    )
    stream: Optional[bool] = Field(
        False,
        description="开启流式返回"
    )
    speed: Annotated[Optional[float], Field(strict=True, ge=0.25, le=4.0)] = Field(
        1.0,
        description="语速控制[0.25-4.0]"
    )
