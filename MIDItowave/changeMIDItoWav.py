# import pretty_midi
# import os
# import subprocess
# from tqdm import tqdm
#
# # === 사용자 설정 ===
# SF2_PATH = "/Users/simjuheun/SoundFonts/FluidR3_GM/FluidR3_GM.sf2"
# MIDI_DIR = "/Volumes/Extreme SSD/lmd_classified/jazz"
# OUTPUT_DIR = "/Volumes/Extreme SSD/lmd_classified/jazzforwav"
# MAX_FILES = 5000  # None 이면 전체 처리
# SAMPLE_RATE = "44100"
#
# # === 준비 ===
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# if not os.path.isfile(SF2_PATH):
#     raise FileNotFoundError(f"❌ 사운드폰트 파일이 존재하지 않음: {SF2_PATH}")
#
# midi_files = [
#     f for f in os.listdir(MIDI_DIR)
#     if f.lower().endswith((".mid", ".midi")) and not f.startswith("._")
# ]
#
# if MAX_FILES:
#     midi_files = midi_files[:MAX_FILES]
#
# print(f"🎧 변환 대상 MIDI 파일 수: {len(midi_files)}")
#
# converted = 0
# failed = []
#
# for filename in tqdm(midi_files, desc="🔄 변환 진행 중"):
#     midi_path = os.path.join(MIDI_DIR, filename)
#     wav_name = filename.rsplit(".", 1)[0] + ".wav"
#     wav_path = os.path.join(OUTPUT_DIR, wav_name)
#
#     if os.path.exists(wav_path):
#         continue  # 이미 변환된 파일 스킵
#
#     try:
#         # MIDI 유효성 검사 및 임시 저장
#         midi = pretty_midi.PrettyMIDI(midi_path)
#         temp_mid = "temp.mid"
#         midi.write(temp_mid)
#
#         # fluidsynth 실행
#         subprocess.run([
#             "fluidsynth",
#             "-ni", SF2_PATH,
#             temp_mid,
#             "-F", wav_path,
#             "-r", SAMPLE_RATE
#         ], check=True)
#
#         converted += 1
#
#     except Exception as e:
#         failed.append((filename, str(e)))
#         print(f"❌ 변환 실패: {filename} - {e}")
#
# # === 결과 출력 ===
# print(f"\n✅ 변환 완료: {converted}개")
# print(f"❌ 실패: {len(failed)}개")
# if failed:
#     with open("failed_conversion.log", "w") as logf:
#         for name, err in failed:
#             logf.write(f"{name}: {err}\n")
#     print("📄 실패 로그: failed_conversion.log 저장됨")



import pretty_midi
import os
import subprocess

# === 사용자 설정 ===
SF2_PATH = "/Users/simjuheun/SoundFonts/FluidR3_GM/FluidR3_GM.sf2"
MIDI_PATH = "/Users/simjuheun/Desktop/Unity/freedom_balloon/Assets/Audio/bgm.mid"
OUTPUT_WAV_PATH = "/Users/simjuheun/Desktop/Unity/freedom_balloon/Assets/Audio/bgm.wav"
SAMPLE_RATE = "44100"

# === 확인 ===
if not os.path.isfile(SF2_PATH):
    raise FileNotFoundError(f"❌ 사운드폰트 파일이 존재하지 않음: {SF2_PATH}")

if not os.path.isfile(MIDI_PATH):
    raise FileNotFoundError(f"❌ MIDI 파일이 존재하지 않음: {MIDI_PATH}")

try:
    # MIDI 유효성 검사 및 임시 저장
    midi = pretty_midi.PrettyMIDI(MIDI_PATH)
    temp_mid = "temp.mid"
    midi.write(temp_mid)

    # fluidsynth 실행
    subprocess.run([
        "fluidsynth",
        "-ni", SF2_PATH,
        temp_mid,
        "-F", OUTPUT_WAV_PATH,
        "-r", SAMPLE_RATE
    ], check=True)

    print(f"✅ 변환 완료: {OUTPUT_WAV_PATH}")

except Exception as e:
    print(f"❌ 변환 실패: {e}")