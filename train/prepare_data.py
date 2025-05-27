import os
import json
from preprocessing.midi_to_chords import extract_chord_sequence

# MIDI 폴더 및 출력 경로 설정
midi_dir = "/Volumes/Extreme SSD/NewDataSet/blues"
output_dir = "./train"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "blues_chords.json")

# MIDI 파일 필터링
all_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid') and not f.startswith('._')]
total_files = len(all_files)
all_sequences = []

# MIDI 처리 루프
for idx, file in enumerate(all_files):
    print(f"🔄 [{idx+1}/{total_files}] Processing {file}...")

    try:
        full_path = os.path.join(midi_dir, file)
        seq = extract_chord_sequence(full_path)
        if len(seq) > 2:
            all_sequences.append(seq)
            print(f"✅ Success ({len(seq)} chords)")
            # ▶ 예시 출력
            preview = " → ".join(seq[:6]) + ("..." if len(seq) > 6 else "")
            print(f"   🎶 Preview: {preview}")
        else:
            print("⚠️ Too short, skipped.")
    except Exception as e:
        print(f"❌ Error: {e}")

# 최종 결과 저장
print(f"\n🎯 Finished! Extracted {len(all_sequences)} chord sequences out of {total_files} files.")
with open(output_path, 'w') as f:
    json.dump(all_sequences, f)
print(f"📁 Saved to: {output_path}")