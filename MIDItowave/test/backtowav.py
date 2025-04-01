import sys
sys.path.append('/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan')
import torch
import numpy as np
import json
import os
import soundfile as sf
from models import Generator

# 경로 설정
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test/0ec264d4f0b5938d9d074e4b252e9d5e.npy"
checkpoint_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/universal/g_02500000"
config_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/universal/config.json"
output_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test/0ec264d4f0b5938d9d074e4b252e9d5e_generated.wav"

# AttrDict 정의
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# Config 로드
with open(config_path) as f:
    config = AttrDict(json.load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = Generator(config).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['generator'])
model.eval()
model.remove_weight_norm()

# Mel 로드
mel = np.load(mel_path)

# ✅ 자동 평균 에너지 보정
mel_mean = mel.mean()
target_mean = 1.0  # 평균이 이 값 이상 되도록 조정
if mel_mean < target_mean:
    scale_factor = target_mean / (mel_mean + 1e-9)
    mel *= scale_factor
    print(f"🔧 평균 에너지 낮음 → 스케일 보정 적용: x{scale_factor:.2f}")
else:
    print(f"✅ 평균 에너지 양호: {mel_mean:.4f}")

# 텐서 변환
mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(device)

# 추론
with torch.no_grad():
    audio = model(mel_tensor).squeeze().cpu().numpy()

# 정규화 후 저장
audio = audio - np.mean(audio)
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))
else:
    print("⚠️ 무음 오디오입니다. 정규화 생략")

sf.write(output_path, audio, config.sampling_rate)

# 출력
print(f"🎧 Audio saved to: {output_path}")
print(f"🔍 MEL shape: {mel.shape} | mean: {mel.mean():.4f}, min: {mel.min():.4f}, max: {mel.max():.4f}")
print(f"🔊 Audio stats → shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}, mean: {audio.mean():.4f}")