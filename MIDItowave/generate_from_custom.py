import sys
sys.path.append("/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan")  # ✅ 먼저 오도록 이동

import torch
import numpy as np
import soundfile as sf
from models import Generator
import json
import os

# ✅ AttrDict 정의
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# [1] 경로 설정
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/user_input/test_user.npy"
output_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/user_input/output_custom.wav"
config_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/config_v3.json"
model_ckpt = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/hifi-gan/g_custom.pth"

# [2] 설정 로드
with open(config_path) as f:
    config = json.load(f)
config = AttrDict(config)

# [3] 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(config).to(device)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.eval()

# [4] Mel 불러오기
mel = np.load(mel_path)
mel = torch.FloatTensor(mel).unsqueeze(0).to(device)  # (1, n_mels, T)

# [5] Wav 생성
with torch.no_grad():
    audio = model(mel).squeeze().cpu().numpy()

# [6] 저장
sf.write(output_path, audio, samplerate=22050)
print(f"✅ Wav 저장 완료: {output_path}")