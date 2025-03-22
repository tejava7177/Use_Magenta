import pretty_midi
import os
import subprocess

# [1] 사운드폰트 경로 (너가 설정한 경로)
SF2_PATH = "/Users/simjuheun/SoundFonts/FluidR3_GM/FluidR3_GM.sf2"

# [2] 변환할 MIDI 폴더와 저장할 WAV 폴더
MIDI_DIR = "/Volumes/Extreme SSD/lmd_classified/jazz"
OUTPUT_DIR = "/Volumes/Extreme SSD/lmd_classified/jazzforwav"
MAX_FILES = 10  # 테스트로 10개만

# [3] 저장 폴더 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

converted = 0

for filename in os.listdir(MIDI_DIR):
    if not filename.lower().endswith((".mid", ".midi")):
        continue

    midi_path = os.path.join(MIDI_DIR, filename)
    wav_path = os.path.join(OUTPUT_DIR, filename.rsplit('.', 1)[0] + ".wav")

    try:
        # MIDI 유효성 검사
        midi = pretty_midi.PrettyMIDI(midi_path)
        temp_mid = "temp.mid"
        midi.write(temp_mid)

        # fluidsynth 실행: temp.mid → wav_path
        subprocess.run([
            "fluidsynth",
            "-ni", SF2_PATH,
            temp_mid,
            "-F", wav_path,
            "-r", "44100"
        ], check=True)

        print(f"✅ 변환 성공: {filename} → {wav_path}")
        converted += 1

        if converted >= MAX_FILES:
            break

    except Exception as e:
        print(f"❌ 변환 실패: {filename} - {e}")