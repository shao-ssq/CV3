import os

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from cosyvoice.utils.file_utils import load_wav
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    model_dir = r'/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B'
    cosyvoice = AsyncCosyVoice2(model_dir, fp16=True)
    spk_id = "003"
    audio_path = "/root/PycharmProjects/CV3/async_cosyvoice/runtime/async_grpc/zero_shot_prompt.wav"
    prompt_text = "希望你以后能够做的比我还好呦。"
    prompt_speech_16k = load_wav(audio_path, 16000)
    cosyvoice.frontend.generate_spk_info(spk_id, prompt_text, prompt_speech_16k, 24000, "A")