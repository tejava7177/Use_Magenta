import os
import numpy as np
import librosa

# 입력/출력 경로 설정
input_wav = "/Volumes/Extreme SSD/lmd_classified/jazzforwav/0ec264d4f0b5938d9d074e4b252e9d5e.wav"
output_dir = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test"
file_id = os.path.splitext(os.path.basename(input_wav))[0]
output_npy = os.path.join(output_dir, f"{file_id}.npy")

os.makedirs(output_dir, exist_ok=True)

# librosa 로딩 및 MEL 추출 (★ dB 변환 생략!)
y, sr = librosa.load(input_wav, sr=22050)
mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=8000  # 🔥 꼭 필요!
)

# 저장
np.save(output_npy, mel_spec)
print(f"✅ MEL 저장 완료 (Universal 모델용): {output_npy}")
print(f"📊 MEL Stats → shape: {mel_spec.shape}, min: {mel_spec.min():.5f}, max: {mel_spec.max():.5f}")