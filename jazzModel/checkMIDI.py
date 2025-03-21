import torch
import random
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

MODEL_PATH = "midi_transformer_20250321_115350.pth"


def build_chord(chord_notes, time_offset, velocity_range=(80, 110)):
    velocity = random.randint(*velocity_range)
    return [[1, note, time_offset, velocity] for note in chord_notes]


seed_sequence = []
# Piano 트랙 (보이싱 + 아르페지오 적용)
seed_sequence += build_chord([60, 64, 67], 0)  # C 코드 기본
seed_sequence += build_chord([72, 76, 79], 120)  # C 코드 옥타브
seed_sequence += build_chord([60, 64, 67], 240)  # 반복
seed_sequence += build_chord([67, 71, 74], 480)  # G 코드
seed_sequence += build_chord([79, 83, 86], 600)  # G 코드 옥타브
seed_sequence += build_chord([65, 69, 72], 960)  # F 코드

# Bass 트랙 (워킹 베이스 스타일 추가)
seed_sequence += build_chord([36], 0, (70, 90))  # C 루트
seed_sequence += build_chord([38], 120, (70, 90))  # 워킹음
seed_sequence += build_chord([40], 240, (70, 90))  # 다른 보이싱
seed_sequence += build_chord([43], 360, (70, 90))  # 베이스 리드
seed_sequence += build_chord([31], 480, (70, 90))  # G 루트
seed_sequence += build_chord([33], 600, (70, 90))  # 워킹음
seed_sequence += build_chord([29], 960, (70, 90))  # F 루트


# Drums 트랙 (스윙 리듬 기반 드럼 추가)
def build_drum_pattern(time_offsets, velocity_range=(50, 90)):
    return [[1, 36, t, random.randint(*velocity_range)] for t in time_offsets]  # Kick Drum


seed_sequence += build_drum_pattern([0, 240, 480, 720, 960])  # 4-beat 패턴 추가


# 모델 로드
class MidiTransformer(torch.nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        self.pos_embedding = torch.nn.Embedding(512, embed_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = torch.nn.Linear(embed_dim, 4)

    def forward(self, x):
        seq_length = x.size(1)
        pos = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.transformer_encoder(x)
        return self.fc_out(x)


# 모델 예측 함수
def generate_notes(model, seed, total_events=2048):
    sequence = seed.copy()
    with torch.no_grad():
        for _ in range(total_events):
            input_tensor = torch.tensor(sequence[-512:], dtype=torch.float32).unsqueeze(0).to(device)
            output = model(input_tensor)
            next_event = output[0, -1].cpu().tolist()
            sequence.append(next_event)
    return sequence


def split_tracks(sequence):
    piano, bass, drums = [], [], []
    for event in sequence:
        etype, pitch, time, velocity = [int(round(x)) for x in event]
        pitch = max(0, min(127, pitch))
        time = max(30, min(240, time))  # ✅ 시간 간격 조정
        velocity = max(60, min(110, velocity))
        if pitch >= 50:
            piano.append((etype, pitch, time, velocity))
        elif pitch < 50 and pitch > 0:
            bass.append((etype, pitch, time, velocity))
        else:
            drums.append((etype, pitch, time, velocity))
    return piano, bass, drums


def save_multitrack_midi(piano, bass, drums, filename="updated_piano_bass.mid"):
    midi = MidiFile()
    piano_track, bass_track, drum_track = MidiTrack(), MidiTrack(), MidiTrack()
    midi.tracks.append(piano_track)
    midi.tracks.append(bass_track)
    midi.tracks.append(drum_track)
    piano_track.append(MetaMessage('track_name', name='Piano', time=0))
    bass_track.append(MetaMessage('track_name', name='Bass', time=0))
    drum_track.append(MetaMessage('track_name', name='Drums', time=0))
    piano_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(100), time=0))

    for etype, pitch, time, velocity in piano:
        msg_type = 'note_on' if etype == 1 else 'note_off'
        piano_track.append(Message(msg_type, channel=0, note=pitch, velocity=velocity, time=time))
    for etype, pitch, time, velocity in bass:
        msg_type = 'note_on' if etype == 1 else 'note_off'
        bass_track.append(Message(msg_type, channel=1, note=pitch, velocity=velocity, time=time))
    for etype, pitch, time, velocity in drums:
        msg_type = 'note_on' if etype == 1 else 'note_off'
        drum_track.append(Message(msg_type, channel=9, note=pitch, velocity=velocity, time=time))
    midi.save(filename)
    print(f"🎹 업데이트된 MIDI 저장 완료: {filename}")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MidiTransformer().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

generated = generate_notes(model, seed_sequence, total_events=2048)
piano_events, bass_events, drum_events = split_tracks(generated)
save_multitrack_midi(piano_events, bass_events, drum_events)
