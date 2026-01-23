import os
import sys
import signal
import atexit

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice3
from cosyvoice.utils.file_utils import load_wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 全局变量用于存储 cosyvoice 实例
cosyvoice_instance = None

def cleanup():
    """清理函数，确保资源被释放"""
    global cosyvoice_instance
    if cosyvoice_instance is not None:
        print("Cleaning up resources...")
        try:
            cosyvoice_instance.shutdown()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        cosyvoice_instance = None

def signal_handler(signum, frame):
    """信号处理器，处理 Ctrl+C 等中断信号"""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)

if __name__ == '__main__':
    # 注册清理函数和信号处理器
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model_dir = r'/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B'
    cosyvoice_instance = AsyncCosyVoice3(model_dir, fp16=True)

    try:
        spk_id = "003"
        audio_path = "/root/PycharmProjects/CV3/async_cosyvoice/runtime/async_grpc/zero_shot_prompt.wav"
        prompt_text = "希望你以后能够做的比我还好呦。"
        prompt_speech_16k = load_wav(audio_path, 16000)
        cosyvoice_instance.frontend.generate_spk_info(spk_id, prompt_text, prompt_speech_16k, 24000, "A")

        print("Available speakers:", cosyvoice_instance.list_available_spks())
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被正确清理
        cleanup()