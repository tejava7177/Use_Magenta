import os
import json
import gzip
import pretty_midi
import numpy as np
from tqdm import tqdm  # ✅ 진행률 확인


# ✅ MIDI 데이터셋 디렉토리 설정
JAZZ_DIR = "/Volumes/Extreme SSD/lmd_classified/jazz"
SAVE_DIR = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel"
JSON_SAVE_PATH = os.path.join(SAVE_DIR, "jazz_dataset_transformer.jsonl.gz")

# ✅ 저장 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)


def transform_midi_to_events(midi_data):
    """🎵 Transformer 학습을 위한 MIDI 데이터를 이벤트 시퀀스로 변환"""
    transformed_data = {}

    for instrument, data in midi_data["track_data"].items():
        note_events = []

        for note in data["melody"]:
            note_events.append({"event": "Note-On", "pitch": note["pitch"], "time": int(note["start_time"]),
                                "velocity": note["velocity"]})
            note_events.append(
                {"event": "Note-Off", "pitch": note["pitch"], "time": int(note["start_time"] + note["duration"])})

        transformed_data[instrument] = {
            "events": sorted(note_events, key=lambda x: x["time"]),
            "velocity": data["velocity"]
        }

    return transformed_data


def extract_midi_features(midi_path):
    """🎷 MIDI 파일에서 Transformer-friendly 데이터를 추출"""
    try:
        if midi_path.startswith("._"):  # ✅ 숨김 파일 무시
            return None

        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # ✅ 트랙이 없으면 제외
        if not midi_data.instruments:
            return None

        # ✅ 템포, 박자, 키 정보
        tempo = max(midi_data.estimate_tempo(), 120)
        time_signature = "4/4" if not midi_data.time_signature_changes else f"{midi_data.time_signature_changes[0].numerator}/{midi_data.time_signature_changes[0].denominator}"
        key = "C Major" if not midi_data.key_signature_changes else midi_data.key_signature_changes[0].key_number

        track_data = {}

        for instrument in midi_data.instruments:
            instrument_name = "Drums" if instrument.is_drum else pretty_midi.program_to_instrument_name(
                instrument.program)

            if instrument_name not in track_data:
                track_data[instrument_name] = {"melody": [], "velocity": {}}

            # ✅ 멜로디 데이터 저장
            for note in instrument.notes:
                track_data[instrument_name]["melody"].append({
                    "pitch": note.pitch,
                    "start_time": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity
                })

            # ✅ velocity 정리
            velocities = [note["velocity"] for note in track_data[instrument_name]["melody"]]
            track_data[instrument_name]["velocity"] = {
                "min": max(10, min(velocities)) if velocities else 10,
                "max": max(velocities) if velocities else 127,
                "avg": sum(velocities) // len(velocities) if velocities else 64
            }

        return {
            "genre": "jazz",
            "tempo": int(tempo),
            "time_signature": time_signature,
            "key": key,
            "instruments": list(track_data.keys()),
            "midi_file": os.path.basename(midi_path),
            "track_data": transform_midi_to_events({"track_data": track_data})
        }

    except Exception as e:
        print(f"❌ 오류 발생: {midi_path} - {e}")
        return None


def process_midi_files():
    """🎶 모든 MIDI 파일을 순차적으로 처리하고, JSONL로 저장"""
    excluded_count = 0
    total_processed = 0

    with gzip.open(JSON_SAVE_PATH, "wt", encoding="utf-8") as f:
        midi_files = [m for m in os.listdir(JAZZ_DIR) if m.endswith(".mid")]

        for midi_file in tqdm(midi_files, desc="🎵 Processing MIDI Files"):
            midi_path = os.path.join(JAZZ_DIR, midi_file)
            features = extract_midi_features(midi_path)

            if features:
                json.dump(features, f)
                f.write("\n")  # ✅ JSONL 형식 (줄바꿈)
                total_processed += 1
            else:
                excluded_count += 1

    print(f"\n✅ JSONL 저장 완료: {JSON_SAVE_PATH}")
    print(f"🚀 처리된 파일 수: {total_processed}, ❌ 제외된 파일 수: {excluded_count}")


# ✅ 실행
process_midi_files()