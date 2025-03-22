import torch
import numpy as np
import json
import librosa
import soundfile as sf
import os

# [0] 경로 설정
BASE_DIR = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/user_input"
jsonl_path = os.path.join(BASE_DIR, "user_input.jsonl")
mel_output_path = os.path.join(BASE_DIR, "generated_user_mel.npy")
wav_output_path = os.path.join(BASE_DIR, "generated_user_output.wav")
model_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/mel_model.pth"

# [1] 사용자 feature 불러오기
with open(jsonl_path, "r") as f:
    user_entry = json.loads(f.readline())

features = user_entry["features"]
x_input = np.array([
    features["tempo"],
    features["duration"],
    features["pitch_mean"],
    features["pitch_min"],
    features["pitch_max"],
    features["rms_energy"],
    features["onset_density"],
    *features["mfcc_mean"]
], dtype=np.float32)

# [2] 모델 정의 (학습 시와 동일 구조)
class ConditionToMelNet(torch.nn.Module):
    def __init__(self, input_dim=20, output_dim=80 * 400):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# [3] 모델 로드 및 예측
input_tensor = torch.tensor(x_input).unsqueeze(0)
model = ConditionToMelNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    mel_output = model(input_tensor).squeeze().numpy()
    mel_output = mel_output.reshape(80, 400)

# [4] Mel-spectrogram 저장
np.save(mel_output_path, mel_output)
print(f"✅ Mel 예측 저장 완료: {mel_output_path}")

# [5] Mel → WAV 변환
mel_power = librosa.db_to_power(mel_output)
wav = librosa.feature.inverse.mel_to_audio(mel_power, sr=22050, n_iter=60)

# [6] WAV 저장
sf.write(wav_output_path, wav, 22050)
print(f"🎧 사용자 입력 기반 생성 음원 저장 완료: {wav_output_path}")