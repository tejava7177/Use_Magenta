# 📄 File: SongMaker_score/utils/midi_to_score.py

from music21 import converter, environment, note
import os

def midi_to_musicxml(midi_path: str, output_path: str):
    us = environment.UserSettings()
    us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    score = converter.parse(midi_path)
    score.makeMeasures(inPlace=True)

    # ✅ 부족한 마디를 rest로 채우기
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            dur = measure.duration.quarterLength
            bar_dur = measure.barDuration.quarterLength
            if dur < bar_dur:
                rest_duration = bar_dur - dur
                measure.append(note.Rest(quarterLength=rest_duration))

    score.write('musicxml', fp=output_path)
    print(f"✅ MusicXML 생성 완료: {output_path}")

# 실행 예시
if __name__ == "__main__":
    midi_file = "/Users/simjuheun/Desktop/myProject/Use_Magenta/SongMaker_score/midi/test.mid"
    xml_output = "/Users/simjuheun/Desktop/myProject/Use_Magenta/SongMaker_score/output/score.musicxml"
    midi_to_musicxml(midi_file, xml_output)