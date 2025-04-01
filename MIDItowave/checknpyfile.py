import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display

# 🎯 분석할 .npy 경로 (사용자 지정)
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/test/0ec264d4f0b5938d9d074e4b252e9d5e.npy"

def analyze_mel(mel_path):
    if not os.path.exists(mel_path):
        print(f"❌ 파일을 찾을 수 없습니다: {mel_path}")
        return

    mel = np.load(mel_path)
    print(f"📂 파일 로드 완료: {mel_path}")
    print(f"📐 Shape: {mel.shape}  (n_mels x T)")
    print(f"📊 값 통계:")
    print(f"   Min:  {mel.min():.6f}")
    print(f"   Max:  {mel.max():.6f}")
    print(f"   Mean: {mel.mean():.6f}")
    print(f"   Std:  {mel.std():.6f}")
    print("Mel T:", mel.shape[1])

    # 🔍 시각화
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.tight_layout()
    plt.show()


# ✅ 실행
if __name__ == "__main__":
    analyze_mel(mel_path)