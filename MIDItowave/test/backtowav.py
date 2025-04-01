import os
import json
import sys
sys.path.append('/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan')

import numpy as np
import torch
import soundfile as sf
from models import Generator

# 경로 설정
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test/0ec264d4f0b5938d9d074e4b252e9d5e.npy"
config_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/config.json"
checkpoint_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/universal/g_02500000"
output_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test/test_universal.wav"

# AttrDict 정의
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# ✅ Config 로드
with open(config_path) as f:
    config = AttrDict(json.load(f))

# ✅ Mel 로드 (★ dB 변환 없음)
mel = np.load(mel_path)
mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # (1, 80, T)

# ✅ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(config).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["generator"])
model.eval()

# ✅ 추론
with torch.no_grad():
    audio = model(mel_tensor.to(device)).squeeze().cpu().numpy()

# ✅ 정규화
audio = audio - np.mean(audio)
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))

# ✅ 저장
sf.write(output_path, audio, config.sampling_rate)

# ✅ 출력
print(f"✅ Universal 모델로 생성된 WAV 저장 완료: {output_path}")
print(f"🔍 Audio shape: {audio.shape} | min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")