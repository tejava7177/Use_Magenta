import torch
import random
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from chord_to_notes import CHORD_TO_NOTES  # 🎹 코드 구성음

# ✅ 모델 경로 및 디바이스 설정
MODEL_PATH = "midi_transformer_20250321_220621.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ✅ 코드 진행 설정 (8마디)
CHORD_SEQUENCE = ["Cmaj7", "Gmaj7", "Fmin", "Cmaj7", "Dmaj7", "Gmaj7", "Fmaj7", "Gmaj7"]
TICKS_PER_BEAT = 480
BPM = 100
TICKS_PER_BAR = TICKS_PER_BEAT * 4  # 4/4 기준

# ✅ 모델 정의 (학습 구조와 동일)
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

# ✅ 시드 시퀀스 생성 함수
def generate_seed(chord_notes, bar_start_tick):
    seed = []
    for i, pitch in enumerate(chord_notes):
        seed.append([
            1,
            pitch,
            bar_start_tick + i * 60,
            random.randint(70, 110)
        ])
    return seed

# ✅ 모델 기반 예측 함수 (각 코드마다 1마디씩 생성)
@torch.no_grad()
def generate_notes_for_chord(model, chord_notes, start_tick):
    seed = generate_seed(chord_notes, start_tick)
    sequence = seed.copy()
    current_time = start_tick

    while current_time < start_tick + TICKS_PER_BAR:
        input_tensor = torch.tensor(sequence[-512:], dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)[0, -1].cpu().tolist()

        etype = int(round(output[0])) % 2
        pitch = int(round(max(40, min(100, output[1]))))
        delay = int(round(max(30, min(240, output[2]))))
        velocity = int(round(max(60, min(127, output[3]))))

        current_time += delay
        if current_time >= start_tick + TICKS_PER_BAR:
            break

        sequence.append([etype, pitch, current_time, velocity])
        # note_off 추가
        sequence.append([0, pitch, current_time + random.randint(30, 180), velocity])

    return sequence

# ✅ 전체 MIDI 저장 함수
def save_midi(events, filename="generated_jazz_style.mid"):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(MetaMessage('track_name', name='Generated Piano', time=0))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(BPM), time=0))
    track.append(Message('program_change', program=0, channel=0, time=0))

    events.sort(key=lambda x: x[2])
    prev_time = 0
    for etype, pitch, abs_time, velocity in events:
        delta = abs_time - prev_time
        msg_type = 'note_on' if etype == 1 else 'note_off'
        track.append(Message(msg_type, channel=0, note=pitch, velocity=velocity, time=delta))
        prev_time = abs_time

    midi.save(filename)
    print(f"🎶 MIDI 저장 완료: {filename}")

# ✅ 메인 실행
def main():
    print("🧠 모델 로딩 중...")
    model = MidiTransformer().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    all_events = []
    for i, chord_name in enumerate(CHORD_SEQUENCE):
        start_tick = i * TICKS_PER_BAR
        chord_notes = CHORD_TO_NOTES.get(chord_name, [60, 64, 67])
        print(f"🎼 코드: {chord_name}, 시작 tick: {start_tick}, 구성음: {chord_notes}")
        events = generate_notes_for_chord(model, chord_notes, start_tick)
        all_events.extend(events)

    save_midi(all_events)

if __name__ == "__main__":
    main()
