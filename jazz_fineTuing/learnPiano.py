import os
import json
import gzip
from mido import MidiFile
from pathlib import Path

def extract_piano_tracks_from_midi(midi_path):
    try:
        midi = MidiFile(midi_path)
        ticks_per_beat = midi.ticks_per_beat
        events = []

        for i, track in enumerate(midi.tracks):
            program = None
            is_piano = False
            abs_time = 0

            for msg in track:
                abs_time += msg.time
                if msg.type == 'program_change':
                    program = msg.program
                    is_piano = (0 <= program <= 7)
                elif msg.type in ['note_on', 'note_off'] and is_piano:
                    events.append({
                        "event": "Note-On" if msg.type == 'note_on' and msg.velocity > 0 else "Note-Off",
                        "pitch": msg.note,
                        "time": abs_time,
                        "velocity": msg.velocity if msg.type == 'note_on' else 0
                    })

            if is_piano and len({e["pitch"] for e in events}) >= 10:
                return {
                    "filename": midi_path.name,
                    "ticks_per_beat": ticks_per_beat,
                    "track_name": track.name if track.name else "Piano",
                    "events": events
                }
        return None
    except Exception as e:
        print(f"❌ 오류: {midi_path.name} → {e}")
        return None

def process_midi_folder(midi_folder, output_jsonl_gz, limit=10000):
    midi_files = list(Path(midi_folder).rglob("*.mid"))[:limit]
    results = []

    for midi_path in midi_files:
        data = extract_piano_tracks_from_midi(midi_path)
        if data:
            results.append(data)
            print(f"🎹 {midi_path.name} → 피아노 트랙 추출 완료")

    with gzip.open(output_jsonl_gz, "wt", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"\n✅ 총 {len(results)}개 파일 저장 완료: {output_jsonl_gz}")

# 예시 사용법
if __name__ == "__main__":
    midi_dir = "/Volumes/Extreme SSD/lmd_classified/jazz"
    output_path = "jazz_piano_dataset.jsonl.gz"
    process_midi_folder(midi_dir, output_path, limit=10000)