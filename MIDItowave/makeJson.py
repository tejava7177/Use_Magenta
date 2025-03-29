import librosa
import librosa.display
import numpy as np
import json
import os
from tqdm import tqdm

# [0] 경로 설정
AUDIO_DIR = "/Volumes/Extreme SSD/lmd_classified/jazzforwav"
OUTPUT_DIR = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave"
MEL_DIR = os.path.join(OUTPUT_DIR, "melspec")
JSONL_PATH = os.path.join(OUTPUT_DIR, "mini_dataset_3200.jsonl")

os.makedirs(MEL_DIR, exist_ok=True)

# [1] ._로 시작하지 않는 .wav 파일 100개 선택
wav_files = sorted([
    f for f in os.listdir(AUDIO_DIR)
    if f.endswith(".wav") and not f.startswith("._")
])[:3200]

jsonl_data = []

for fname in tqdm(wav_files, desc="🎧 Processing WAV files"):
    try:
        input_wav = os.path.join(AUDIO_DIR, fname)
        file_id = os.path.splitext(fname)[0]
        mel_path = os.path.join(MEL_DIR, f"{file_id}.npy")

        y, sr = librosa.load(input_wav, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_density = len(onset_times) / duration

        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'),
                         fmax=librosa.note_to_hz('C6'), sr=sr)
        pitch_mean = float(np.mean(f0))
        pitch_min = float(np.min(f0))
        pitch_max = float(np.max(f0))

        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(mel_path, mel_db)

        json_obj = {
            "id": file_id,
            "genre": "jazz",
            "audio_path": input_wav,
            "mel_path": mel_path,
            "features": {
                "tempo": float(tempo),
                "duration": duration,
                "pitch_mean": pitch_mean,
                "pitch_min": pitch_min,
                "pitch_max": pitch_max,
                "rms_energy": rms_mean,
                "onset_density": onset_density,
                "mfcc_mean": mfcc_mean
            }
        }

        jsonl_data.append(json_obj)

    except Exception as e:
        print(f"❌ {fname} 처리 중 오류 발생: {e}")

# [2] JSONL 파일로 저장
with open(JSONL_PATH, "w") as f:
    for item in jsonl_data:
        json.dump(item, f)
        f.write("\n")

print(f"\n✅ 총 {len(jsonl_data)}개 파일 처리 완료!")
print(f"📁 JSONL 저장: {JSONL_PATH}")
print(f"📁 Mel 저장 경로: {MEL_DIR}")