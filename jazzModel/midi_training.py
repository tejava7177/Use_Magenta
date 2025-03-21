import json
import gzip
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ✅ 경로 설정
JSONL_FILE = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/jazz_dataset_transformer.jsonl.gz"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f"midi_transformer_{timestamp}.pth"

# ✅ JSONL 로드 함수 (1000개 제한)
def load_jsonl(jsonl_path, max_items=1000):
    data = []
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            data.append(json.loads(line))
    return data

# ✅ Dataset 클래스
class MidiDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_length = 512

    def tokenize(self, track_data):
        return [
            [1 if event["event"] == "Note-On" else 2, event["pitch"], round(event["time"]), event.get("velocity", 0)]
            for details in track_data.values()
            for event in details["events"]
        ]

    def pad_sequence(self, sequence):
        sequence = sequence[:self.max_length]
        pad_len = self.max_length - len(sequence)
        if pad_len > 0:
            sequence.extend([[0, 0, 0, 0]] * pad_len)
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenize(self.data[idx]["track_data"])
        return self.pad_sequence(tokens)

# ✅ Transformer 모델 정의
class MidiTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 4)

    def forward(self, x):
        seq_length = x.size(1)
        pos = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.transformer_encoder(x)
        return self.fc_out(x)

# ✅ 학습 루프
def train(model, dataloader, criterion, optimizer, epochs=10, device="mps"):
    model.to(device)
    model.train()
    print(f"🧠 디바이스: {device}")

    for epoch in range(epochs):
        total_loss = 0
        print(f"🟢 Epoch {epoch + 1}/{epochs} 시작")
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, 4), batch.view(-1, 4))

            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ 손실 값 비정상, 학습 중단")
                return

            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

        avg_loss = total_loss / len(dataloader)
        print(f"🟣 Epoch [{epoch + 1}/{epochs}] - 평균 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 최종 모델 저장됨: {MODEL_PATH}")

# ✅ 메인 실행
if __name__ == "__main__":
    print("📂 JSONL 데이터 로드 중 (1000개만)")
    json_data = load_jsonl(JSONL_FILE, max_items=1000)

    print("🎵 DataLoader 구성 중")
    dataset = MidiDataset(json_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    print("🧱 모델 생성")
    model = MidiTransformer()
    criterion = nn.SmoothL1Loss()  # ✅ 더 안정적인 손실 함수 사용
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 학습 시작")
    train(model, dataloader, criterion, optimizer)
    print("🎉 학습 완료!")