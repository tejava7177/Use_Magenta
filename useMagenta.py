# from magenta.models.music_vae import configs
# from magenta.models.music_vae.trained_model import TrainedModel
# import note_seq
#
# # ✅ MusicVAE 모델 로드 (재즈 스타일 가능)
# music_vae = TrainedModel(
#     configs.CONFIG_MAP['hierarchical'],  # 코드 진행에 맞는 멜로디
#     batch_size=4,
#     checkpoint_dir_or_path='/path/to/music_vae/model'
# )
#
# # ✅ 코드 진행을 바탕으로 시퀀스 생성
# chord_progression = ["Cmaj7", "Gmaj7", "Fmin7", "Dmin7"]
# input_sequence = note_seq.midi_file_to_note_sequence("input_chords.mid")
#
# # ✅ MusicVAE 멜로디 & 화성 생성
# generated_sequence = music_vae.sample(n=1, length=32, temperature=0.8)[0]
#
# # ✅ GrooVAE로 드럼 트랙 추가
# groove_sequence = groovae.generate(input_sequence, num_outputs=1, temperature=1.0)
#
# # ✅ MIDI 저장
# note_seq.sequence_proto_to_midi_file(generated_sequence, "generated_jazz.mid")
# print("🎼 재즈 스타일 MIDI 생성 완료!")
