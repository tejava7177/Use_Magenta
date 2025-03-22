import librosa
import numpy as np
import json
import os

# [0] 설정
input_wav = "/Users/simjuheun/Desktop/test.wav"
file_id = "test_user"
output_dir = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/user_input"
os.makedirs(output_dir, exist_ok=True)

mel_path = os.path.join(output_dir, f"{file_id}.npy")
jsonl_path = os.path.join(output_dir, "user_input.jsonl")

# [1] 오디오 로드
y, sr = librosa.load(input_wav, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# [2] Feature 추출
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_density = len(onset_times) / duration

f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C6'), sr=sr)
pitch_mean = float(np.mean(f0))
pitch_min = float(np.min(f0))
pitch_max = float(np.max(f0))

rms = librosa.feature.rms(y=y)[0]
rms_mean = float(np.mean(rms))

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1).tolist()

# [3] Mel-spectrogram 저장
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
np.save(mel_path, mel_db)

# [4] JSONL 생성
json_obj = {
    "id": file_id,
    "genre": "unknown",
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

with open(jsonl_path, "w") as f:
    json.dump(json_obj, f)
    f.write("\n")

print(f"✅ 사용자 입력 분석 완료! → {jsonl_path}")
print(f"🎵 Mel 저장: {mel_path}")