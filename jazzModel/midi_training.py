import json
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os

# ✅ JSONL 파일 경로
JSONL_FILE = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/jazz_dataset_transformer.jsonl.gz"

# ✅ 모델 저장 경로
MODEL_PATH = "midi_transformer.pth"

# ✅ JSONL 데이터를 한 번만 로드하고 캐싱 (전역 변수)
_cached_json_data = None


def load_jsonl(jsonl_path):
    """📂 JSONL 데이터를 한 번만 로드하고 캐싱"""
    global _cached_json_data
    if _cached_json_data is not None:
        return _cached_json_data  # ✅ 이미 로드된 데이터 반환

    data = []

    # ✅ 전체 줄 개수 확인 (진행률 표시용)
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # ✅ JSONL 파일 로드
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f, tqdm(total=total_lines, desc="📂 Loading JSONL Data",
                                                                  unit=" lines") as pbar:
        for line in f:
            data.append(json.loads(line))
            pbar.update(1)

    _cached_json_data = data  # ✅ 캐싱하여 이후 호출 시 재사용
    return data


class MidiDataset(Dataset):
    """🎵 MIDI 데이터를 Transformer 학습용 Dataset으로 변환"""

    def __init__(self, data):
        self.data = data
        self.max_length = 512  # ✅ Transformer 입력 시퀀스 길이 제한

    def tokenize(self, track_data):
        """✅ MIDI 데이터를 Transformer 입력 형태로 변환 (Tokenize)"""
        return [
            [1 if event["event"] == "Note-On" else 2, event["pitch"], round(event["time"]), event.get("velocity", 0)]
            for details in track_data.values()
            for event in details["events"]
        ]

    def pad_sequence(self, sequence):
        """✅ 시퀀스 길이를 self.max_length에 맞게 패딩"""
        sequence = sequence[:self.max_length]  # 길이 초과 부분 자르기
        pad_len = self.max_length - len(sequence)
        if pad_len > 0:
            sequence.extend([[0, 0, 0, 0]] * pad_len)  # 패딩 추가
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenize(item["track_data"])
        tokens = self.pad_sequence(tokens)
        return tokens


class MidiTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(MidiTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # ✅ 가변적인 Positional Encoding

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, 4)

    def forward(self, x):
        seq_length = x.size(1)
        pos = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)  # ✅ 가변적인 Positional Encoding 적용
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x


def train(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output.view(-1, 4), batch.view(-1, 4))  # ✅ 손실 계산 변경
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()  # ✅ detach() 사용하여 그래프 분리

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")

    # ✅ 학습된 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 모델이 {MODEL_PATH}에 저장되었습니다!")


def load_model(model, model_path="midi_transformer.pth", device="cpu"):
    """✅ 저장된 모델을 불러오는 함수"""
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 저장된 모델 {model_path}을 불러왔습니다!")
    else:
        print("🚀 새 모델을 학습합니다!")


if __name__ == "__main__":
    print("📂 JSONL 데이터 로드 시작...")
    json_data = load_jsonl(JSONL_FILE)

    print("🎵 데이터 전처리 및 DataLoader 생성...")
    dataset = MidiDataset(json_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    print("✅ Transformer 모델 초기화 완료!")
    model = MidiTransformer()
    criterion = nn.CrossEntropyLoss()  # ✅ MSELoss()도 가능
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 기존 모델 불러오기 (있으면 이어서 학습)
    load_model(model, MODEL_PATH)

    print("🚀 모델 학습 시작...")
    train(model, dataloader, criterion, optimizer)

    print("🎶 Transformer 모델 학습 완료! 🚀")