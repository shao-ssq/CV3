import os
import sys
import signal
import atexit

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice3
from cosyvoice.utils.file_utils import load_wav
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
        spk_ids = ["cwl01","女1","施压","柔和","男1","男2","男3"]
        audio_paths = ["/root/PycharmProjects/CV3/async_cosyvoice/wav/cwl01.wav","/root/PycharmProjects/CV3/async_cosyvoice/wav/女1.wav",
                      "/root/PycharmProjects/CV3/async_cosyvoice/wav/施压.wav","/root/PycharmProjects/CV3/async_cosyvoice/wav/柔和.wav",
                      "/root/PycharmProjects/CV3/async_cosyvoice/wav/男1.wav","/root/PycharmProjects/CV3/async_cosyvoice/wav/男2.wav",
                      "/root/PycharmProjects/CV3/async_cosyvoice/wav/男3.wav"]
        prompt_texts = ["我们会稍后确认你的还款状态，如果还是逾期我们会再次联系你。","您好，刘先生，这边是微粒贷贷后的，那您这个钱怎么还没有处理进来呢刘先生？你有钱了你会还的，那你什么时候有钱？",
                      "刚你也说了，你愿意去承担这个责任，再重复一遍，不可能说再给到你更多的时间了，现在要么您这边的话。","您没有发工资跟您没有还款有什么关系呢？那您不会找您的家人和朋友这边去周转一下吗？看他们能不能帮助到您呀？",
                      "那现在也是需要您去进行一个偿还的吧，不可能说啊这个钱借出去了，就想着什么东西对自己重要。","这些后果你都能承担得了是吧？没有任何还款意愿，我们凭什么相信你？你口头上面说你没钱，你就没钱了？",
                      "逾期时间不按我们银行要求进行处理，我们上报您个人征信，那您的这个全款就会面临一个结清的情况。"]
        for spk_id,prompt_text, audio_path in zip(spk_ids, prompt_texts, audio_paths):
            prompt_speech_16k = load_wav(audio_path, 16000)
            cosyvoice_instance.frontend.generate_spk_info(spk_id, "You are a helpful assistant.<|endofprompt|>" + prompt_text, prompt_speech_16k, 24000, "A")

        print("Available speakers:", cosyvoice_instance.list_available_spks())
        print("Done!")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被正确清理
        cleanup()
