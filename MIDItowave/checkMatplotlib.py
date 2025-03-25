import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# 예시로 사용할 파일
wav_path = "/Users/simjuheun/Music/Music/Media.localized/Music/Unknown Artist/Unknown Album/01fd2c93d1267abc21857d6a62996925_hifigan_ver.wav"

# 1. 로딩
y, sr = librosa.load(wav_path, sr=None)  # 원본 샘플레이트 유지
duration = librosa.get_duration(y=y, sr=sr)
print(f"Sample Rate: {sr}, Duration: {duration:.2f} sec")

# 2. 파형 시각화
plt.figure(figsize=(12, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.show()

# 3. 멜 스펙트로그램
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()

# 4. 템포 / 비트 추정
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated Tempo: {tempo:.2f} BPM")

# 5. onset 감지
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
print(f"Detected Onsets (seconds): {onset_times}")