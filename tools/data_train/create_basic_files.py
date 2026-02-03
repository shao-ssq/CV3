import os
from pathlib import Path

# ==================== 配置区域 ====================
RAW_DATA_DIR = "/root/PycharmProjects/CV3/tools/data_train/data"  # 音频文件目录
TRANSCRIPT_FILE = "/path/to/transcripts.txt"  # 文本标注文件
SPEAKER_ID = "xijun"  # 说话人ID（单说话人时固定）
OUTPUT_DIR = "/root/PycharmProjects/CV3/tools/data_train/data/" + SPEAKER_ID

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 读取文本标注 ====================
# 假设transcripts.txt格式为：filename|text
# 例如：audio_001.wav|你好世界
transcripts = {}
if os.path.exists(TRANSCRIPT_FILE):
    with open(TRANSCRIPT_FILE, "r", encoding="utf8") as f:
        for line in f:
            if "|" in line:
                filename, text = line.strip().split("|", 1)
                # 去掉文件扩展名作为key
                key = Path(filename).stem
                transcripts[key] = text

# ==================== 扫描音频文件 ====================
wav_scp = []
text_list = []
utt2spk_list = []

audio_files = sorted(Path(RAW_DATA_DIR).glob("*.wav"))
print(f"找到 {len(audio_files)} 个音频文件")

for idx, audio_file in enumerate(audio_files, start=1):
    # 生成utterance_id（单说话人时使用简单编号）
    utt_id = f"utt{idx:05d}"  # utt00001, utt00002, ...

    # wav.scp - 使用绝对路径
    wav_scp.append(f"{utt_id} {audio_file.absolute()}\n")

    # utt2spk - 单说话人时都映射到同一个speaker
    utt2spk_list.append(f"{utt_id} {SPEAKER_ID}\n")

    # text - 从标注文件读取，或使用占位符
    audio_stem = audio_file.stem
    if audio_stem in transcripts:
        text_content = transcripts[audio_stem]
    else:
        # 如果没有标注，使用占位符（需要手动补充）
        text_content = f"[需要补充文本: {audio_file.name}]"

    text_list.append(f"{utt_id} {text_content}\n")

# ==================== 写入文件 ====================
with open(f"{OUTPUT_DIR}/wav.scp", "w", encoding="utf8") as f:
    f.writelines(wav_scp)

with open(f"{OUTPUT_DIR}/text", "w", encoding="utf8") as f:
    f.writelines(text_list)

with open(f"{OUTPUT_DIR}/utt2spk", "w", encoding="utf8") as f:
    f.writelines(utt2spk_list)

print(f"\n生成文件完成：")
print(f"  - {OUTPUT_DIR}/wav.scp ({len(wav_scp)} 条)")
print(f"  - {OUTPUT_DIR}/text ({len(text_list)} 条)")
print(f"  - {OUTPUT_DIR}/utt2spk ({len(utt2spk_list)} 条)")
print(f"  - 说话人: {SPEAKER_ID}")

# 检查是否有缺少文本标注的
missing_text = [line for line in text_list if "[需要补充文本:" in line]
if missing_text:
    print(f"\n⚠️  警告：{len(missing_text)} 个音频缺少文本标注，请手动补充")
