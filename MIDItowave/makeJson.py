import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# [1] 사용자 입력 파일
input_wav = "/Users/simjuheun/Desktop/test.wav"
y, sr = librosa.load(input_wav, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# [2] 템포 & 비트
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo = float(np.atleast_1d(tempo)[0])  # tempo is sometimes array

# [3] Onset 감지 → 밀도 계산
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_density = len(onset_times) / duration

# [4] Pitch 추정
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C6'), sr=sr)
pitch_mean = float(np.mean(f0))
pitch_min = float(np.min(f0))
pitch_max = float(np.max(f0))

# [5] RMS 에너지 (볼륨 느낌)
rms = librosa.feature.rms(y=y)[0]
rms_mean = float(np.mean(rms))

# [6] Mel Spectrogram (for embedding later)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

# [7] 시각화 (선택)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()

# [8] JSON 저장
feature_dict = {
    "tempo": tempo,
    "duration": duration,
    "onset_density": onset_density,
    "pitch_mean": pitch_mean,
    "pitch_min": pitch_min,
    "pitch_max": pitch_max,
    "rms_energy": rms_mean
}

output_path = "/Users/simjuheun/Desktop/test_audio_features.json"
with open(output_path, "w") as f:
    json.dump(feature_dict, f, indent=4)

print(f"✅ 분석 완료! → {output_path}")