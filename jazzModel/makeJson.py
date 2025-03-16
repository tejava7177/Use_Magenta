import os
import json
import gzip
import pretty_midi

# ✅ MIDI 데이터셋 디렉토리 설정
JAZZ_DIR = "/Volumes/Extreme SSD/lmd_classified/jazz"
SAVE_DIR = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel"

# ✅ 저장 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

def transform_midi_to_events(midi_data):
    """🎵 Transformer 학습을 위한 MIDI 데이터를 이벤트 시퀀스로 변환"""
    transformed_data = {}

    for instrument, data in midi_data["track_data"].items():
        note_events = []

        for note in data["melody"]:
            start_time = int(round(note["start_time"] * 100))  # ✅ 100ms 단위 변환
            end_time = int(round((note["start_time"] + note["duration"]) * 100))

            note_events.append({"event": "Note-On", "pitch": note["pitch"], "time": start_time, "velocity": note["velocity"]})
            note_events.append({"event": "Note-Off", "pitch": note["pitch"], "time": end_time})

        transformed_data[instrument] = {
            "events": sorted(note_events, key=lambda x: x["time"]),  # ✅ 시간 순서 정렬
            "velocity": data["velocity"]  # ✅ velocity 정보 유지
        }

    return transformed_data

def extract_midi_features(midi_path):
    """🎷 MIDI 파일에서 Transformer-friendly 데이터를 추출"""
    try:
        if midi_path.startswith("._"):  # ✅ 숨김 파일 무시
            return None

        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # ✅ 템포, 박자, 키 정보
        tempo = max(midi_data.estimate_tempo(), 120)
        time_signature = "4/4" if not midi_data.time_signature_changes else f"{midi_data.time_signature_changes[0].numerator}/{midi_data.time_signature_changes[0].denominator}"
        key = "C Major" if not midi_data.key_signature_changes else midi_data.key_signature_changes[0].key_number

        track_data = {}

        for instrument in midi_data.instruments:
            instrument_name = "Drums" if instrument.is_drum else pretty_midi.program_to_instrument_name(instrument.program)

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

            # ✅ velocity 정리 (최소값 보정, 대표값 저장)
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
        #print(f"⚠️ 오류 발생: {midi_path} - {e}")
        return None

# ✅ JSON 변환 및 저장 (전체 데이터셋 학습)
def process_midi_files():
    midi_data_list = []

    for midi_file in os.listdir(JAZZ_DIR):
        if midi_file.endswith(".mid"):
            midi_path = os.path.join(JAZZ_DIR, midi_file)
            features = extract_midi_features(midi_path)
            if features:
                midi_data_list.append(features)
                print(f"📂 JSON 데이터 추가됨: {midi_file}")


    return midi_data_list

# ✅ 압축 JSON 저장 (.json.gz로 파일 크기 줄이기)
json_path = os.path.join(SAVE_DIR, "jazz_dataset_transformer.json.gz")

with gzip.open(json_path, "wt", encoding="utf-8") as f:
    json.dump({"jazz": process_midi_files(max_files=5)}, f, indent=4)

print(f"✅ Transformer JSON 데이터 저장 완료: {json_path}")