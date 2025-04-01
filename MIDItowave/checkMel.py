import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import json
import os
import sys

# ✅ 모델 임포트를 위한 경로 추가 (수정 필요)
sys.path.append("/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan")
from models import Generator

# ✅ AttrDict 정의
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# ✅ 설정
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/melspec/2a863c15e2a9f0d56d4071ff9a38a130.npy"
file_id = os.path.splitext(os.path.basename(mel_path))[0]
output_dir = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/result"
os.makedirs(output_dir, exist_ok=True)

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024

# ✅ mel 불러오기
print(f"📥 Loading mel-spectrogram: {mel_path}")
mel_db = np.load(mel_path)
print(f"✅ Loaded mel shape: {mel_db.shape}")

# ✅ 시각화
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram (dB)")
plt.tight_layout()
plt.show()

# ✅ dB → Power
mel = librosa.db_to_power(mel_db)

# ✅ Griffin-Lim
print("🔊 Generating waveform with Griffin-Lim...")
wav_griffin = librosa.feature.inverse.mel_to_audio(
    mel, sr=sr,
    n_fft=n_fft, hop_length=hop_length,
    win_length=win_length, n_iter=60
)

# ✅ 저장
griffin_output_path = os.path.join(output_dir, f"{file_id}_griffin_ver.wav")
sf.write(griffin_output_path, wav_griffin, sr)
print(f"✅ Griffin-Lim WAV 저장 완료: {griffin_output_path}")

# ✅ [2] HiFi-GAN 변환
print("🤖 Generating waveform with HiFi-GAN...")

# 설정 불러오기
config_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/config_v3.json"
model_ckpt = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/g_custom_e10.pth"

with open(config_path) as f:
    config = json.load(f)
config = AttrDict(config)

# 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(config).to(device)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.eval()

# mel 입력 형태 맞추기
mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).to(device)

# 생성
with torch.no_grad():
    wav_hifigan = model(mel_tensor).squeeze().cpu().numpy()

hifigan_output_path = f"{file_id}_hifigan_ver.wav"
sf.write(hifigan_output_path, wav_hifigan, sr)
print(f"✅ HiFi-GAN WAV 저장 완료: {hifigan_output_path}")