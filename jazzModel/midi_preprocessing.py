import json
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm  # ✅ 진행률 표시

# ✅ 데이터셋 경로 설정
JSONL_FILE = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/jazz_dataset_transformer.jsonl.gz"


class MidiDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []

        # ✅ JSONL 파일의 총 줄 수를 미리 계산 (진행률 정확도 개선)
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)  # ✅ 전체 줄 개수 확인

        # ✅ tqdm에 전체 개수 지정하여 진행률 표시
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f, tqdm(total=total_lines, desc="📂 Loading JSONL Data",
                                                                      unit=" lines") as pbar:
            for line in f:
                self.data.append(json.loads(line))
                pbar.update(1)  # ✅ 1줄씩 업데이트

        self.max_length = 512  # ✅ Transformer 입력 길이 제한

    def tokenize(self, track_data):
        """
        ✅ MIDI 데이터를 Transformer 입력 형태로 변환 (Tokenize)
        """
        tokens = []
        for instrument, details in track_data.items():
            for event in details["events"]:
                event_type = 1 if event["event"] == "Note-On" else 2
                tokens.append([
                    event_type,  # 1 = Note-On, 2 = Note-Off
                    event["pitch"],
                    event["time"],
                    event.get("velocity", 0)
                ])
        return tokens

    def pad_sequence(self, sequence):
        """
        ✅ 시퀀스 길이를 self.max_length에 맞게 패딩
        """
        if len(sequence) >= self.max_length:
            return sequence[:self.max_length]
        else:
            pad_len = self.max_length - len(sequence)
            return sequence + [[0, 0, 0, 0]] * pad_len  # 패딩 값 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenize(item["track_data"])
        tokens = self.pad_sequence(tokens)
        return torch.tensor(tokens, dtype=torch.long)


# ✅ 데이터 로드
dataset = MidiDataset(JSONL_FILE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ 진행 상황 확인 (tqdm 추가)
for batch in tqdm(dataloader, desc="🚀 Processing DataLoader Batches", unit=" batch"):
    print(batch.shape)  # ✅ Expected: (32, 512, 4)
    break  # 첫 번째 배치만 확인