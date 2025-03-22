import numpy as np
import librosa
import soundfile as sf
import os

# [0] 멜 파일 경로 지정
mel_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/melspec/74c18b5d748c9d4ceb8983d99311b2ca.npy"
file_id = os.path.splitext(os.path.basename(mel_path))[0]
output_path = f"reconstructed_{file_id}.wav"

# [1] Mel-spectrogram 로드
print(f"📥 Loading mel-spectrogram: {mel_path}")
mel_db = np.load(mel_path)
print(f"✅ Loaded mel shape: {mel_db.shape}")

# [2] dB scale → power scale
print("🔄 Converting dB to power scale...")
mel = librosa.db_to_power(mel_db)

# [3] Griffin-Lim 알고리즘으로 waveform 복원
print("🔊 Reconstructing waveform using Griffin-Lim...")
sr = 22050
wav = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_iter=60)
print(f"✅ Reconstructed waveform shape: {wav.shape}")

# [4] .wav 파일 저장
sf.write(output_path, wav, sr)
print(f"💾 Saved to: {output_path}")