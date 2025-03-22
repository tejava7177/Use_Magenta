from pydub import AudioSegment
import os
import random

# [1] 사용자 입력 파일 (10초 베이스)
input_path = "/Users/simjuheun/Desktop/test.wav"
input_segment = AudioSegment.from_file(input_path)[:10_000]  # 앞 10초

# [2] 기존 WAV 데이터셋에서 샘플 5개 추출
WAV_DIR = "/Volumes/Extreme SSD/lmd_classified/jazzforwav"
wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav") and not f.startswith("._")]
wav_paths = [os.path.join(WAV_DIR, f) for f in wav_files]

# 랜덤으로 5개 고르기
chosen_segments = [AudioSegment.from_file(f)[:10_000] for f in random.sample(wav_paths, 5)]

# [3] 이어붙이기 (총 약 60초)
final_track = input_segment
for seg in chosen_segments:
    final_track += seg

# [4] 저장
output_path = "/Users/simjuheun/Desktop/demo_backingtrack.wav"
final_track.export(output_path, format="wav")
print(f"✅ 1분짜리 백킹트랙 생성 완료! → {output_path}")