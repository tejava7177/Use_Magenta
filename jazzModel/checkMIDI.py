from mido import MidiFile
from collections import Counter

def analyze_midi(filepath):
    midi = MidiFile(filepath)
    ticks_per_beat = midi.ticks_per_beat

    print(f"🎵 MIDI 파일: {filepath}")
    print(f"🕒 Ticks per beat: {ticks_per_beat}")
    print(f"🎼 트랙 수: {len(midi.tracks)}\n")

    total_pitch_counter = Counter()

    for i, track in enumerate(midi.tracks):
        note_count = 0
        velocities = []
        pitch_range = []
        pitch_counter = Counter()
        track_ticks = 0

        for msg in track:
            track_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count += 1
                velocities.append(msg.velocity)
                pitch_range.append(msg.note)
                pitch_counter[msg.note] += 1
                total_pitch_counter[msg.note] += 1

        seconds = midi.length if i == 0 else track_ticks / (ticks_per_beat * 2)

        print(f"🎹 트랙 {i + 1}: {track.name if track.name else '(이름 없음)'}")
        print(f" - 노트 수: {note_count}")
        print(f" - 트랙 길이 (ticks): {track_ticks}")
        print(f" - 추정 길이 (sec): {seconds:.2f}초")
        if velocities:
            print(f" - 평균 벨로시티: {sum(velocities)/len(velocities):.1f}")
        if pitch_range:
            print(f" - 피치 범위: {min(pitch_range)} ~ {max(pitch_range)}")
            most_common = pitch_counter.most_common(5)
            print(f" - 상위 피치 (Top 5): {most_common}")
        print()

    print(f"🧮 전체 재생 시간 (mido 기준): {midi.length:.2f}초")

    # 전체 피치 통계 요약
    if total_pitch_counter:
        print("\n📊 전체 피치 발생 통계 (Top 10):")
        for pitch, count in total_pitch_counter.most_common(10):
            print(f" - 피치 {pitch}: {count}회")

def main():
    midi_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/generated_jazz_style.mid"
    analyze_midi(midi_path)

if __name__ == "__main__":
    main()