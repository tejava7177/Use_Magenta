import os
import json
import gzip
import pretty_midi

# ✅ MIDI 데이터셋 디렉토리 설정
JAZZ_DIR = "/Volumes/Extreme SSD/lmd_classified/jazz"
SAVE_DIR = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel"

# ✅ 저장 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

def optimize_midi_features(midi_data):
    """🎶 JSON 데이터 최적화: 중복 데이터 제거 및 압축"""
    optimized_data = {}

    for instrument, data in midi_data["track_data"].items():
        # ✅ velocity 정리 (중복된 값이 많으면 대표값만 저장)
        unique_velocities = list(set(note["velocity"] for note in data["melody"]))
        unique_velocities = [max(10, v) for v in unique_velocities]  # 최소값 보정

        if len(unique_velocities) > 5:
            velocity_summary = {
                "min": min(unique_velocities),
                "max": max(unique_velocities),
                "avg": sum(unique_velocities) // len(unique_velocities)
            }
        else:
            velocity_summary = unique_velocities  # 그대로 유지

        # ✅ 코드 진행 압축 (반복되는 코드가 많으면 "횟수" 저장)
        chord_sequences = []
        prev_chord = None
        repeat_count = 1

        for chord in data["chords"]:
            if prev_chord and prev_chord["chord"] == chord["chord"]:
                repeat_count += 1
            else:
                if prev_chord:
                    chord_sequences.append({"chord": prev_chord["chord"], "repeat": repeat_count})
                repeat_count = 1
                prev_chord = chord

        # 마지막 코드 추가
        if prev_chord:
            chord_sequences.append({"chord": prev_chord["chord"], "repeat": repeat_count})

        optimized_data[instrument] = {
            "melody": [
                {
                    "pitch": note["pitch"],
                    "time": round(note["start_time"], 2),
                    "d": round(note["duration"], 2)
                }
                for note in data["melody"][:10]  # 샘플 10개 저장 (학습용)
            ],
            "chords": chord_sequences[:10],  # 샘플 10개 유지
            "velocity": velocity_summary
        }

        # ✅ playing_style이 존재하면 추가
        if data["playing_style"]:
            optimized_data[instrument]["playing_style"] = data["playing_style"]

    return optimized_data

def extract_midi_features(midi_path):
    """🎷 MIDI 파일에서 최적화된 악기별 데이터를 추출"""
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
                track_data[instrument_name] = {"melody": [], "chords": [], "playing_style": {}}

            # ✅ 멜로디 데이터 저장
            for note in instrument.notes:
                track_data[instrument_name]["melody"].append({
                    "pitch": note.pitch,
                    "start_time": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity
                })

            # ✅ 코드 진행 저장 (3개 이상의 음이 동시에 연주될 경우 코드로 인식)
            chord_buffer = []
            for note in instrument.notes:
                chord_buffer.append(note.pitch)
                if len(chord_buffer) >= 3:
                    track_data[instrument_name]["chords"].append({
                        "chord": sorted(set(chord_buffer))
                    })
                    chord_buffer = []

            # ✅ 연주 스타일 분석 (기타와 베이스)
            if "Guitar" in instrument_name or "Bass" in instrument_name:
                track_data[instrument_name]["playing_style"] = {
                    "bending": any(note.pitch_bend for note in instrument.pitch_bends),
                    "hammer_on": any(note.start < 0.1 for note in instrument.notes),
                    "pull_off": any(note.end > 1.5 for note in instrument.notes),
                }

        return {
            "genre": "jazz",
            "tempo": int(tempo),
            "time_signature": time_signature,
            "key": key,
            "instruments": list(track_data.keys()),
            "midi_file": os.path.basename(midi_path),
            "track_data": optimize_midi_features({"track_data": track_data})
        }

    except Exception as e:
        print(f"⚠️ 오류 발생: {midi_path} - {e}")
        return None

# ✅ JSON 변환 및 저장 (압축된 JSON 파일 생성)
def process_midi_files():
    midi_data_list = []

    for midi_file in os.listdir(JAZZ_DIR):
        if midi_file.endswith(".mid"):
            midi_path = os.path.join(JAZZ_DIR, midi_file)
            features = extract_midi_features(midi_path)
            if features:
                midi_data_list.append(features)
                print(f"📂 JSON 데이터 추가됨: {midi_file}")

            if len(midi_data_list) >= 20:  # ✅ 20개까지만 저장
                break

    return midi_data_list

# ✅ 압축 JSON 저장 (.json.gz로 파일 크기 줄이기)
json_path = os.path.join(SAVE_DIR, "jazz_dataset.json.gz")

with gzip.open(json_path, "wt", encoding="utf-8") as f:
    json.dump({"jazz": process_midi_files()}, f, indent=4)

print(f"✅ JSON 데이터 저장 완료: {json_path}")