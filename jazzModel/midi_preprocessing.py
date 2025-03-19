# midi_preprocessing.py
import json
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ✅ JSONL 파일 경로
JSONL_FILE = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/jazz_dataset_transformer.jsonl.gz"

# ✅ JSONL 데이터를 한 번만 로드하고 캐싱 (전역 변수 사용)
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
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f, tqdm(total=total_lines, desc="📂 Loading JSONL Data", unit=" lines") as pbar:
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
        """
        ✅ MIDI 데이터를 Transformer 입력 형태로 변환 (Tokenize)
        """
        return [
            [1 if event["event"] == "Note-On" else 2, event["pitch"], round(event["time"]), event.get("velocity", 0)]
            for details in track_data.values()
            for event in details["events"]
        ]

    def pad_sequence(self, sequence):
        """
        ✅ 시퀀스 길이를 self.max_length에 맞게 패딩
        """
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

# ✅ JSONL 데이터를 한 번만 로드하여 미리 캐싱
json_data = load_jsonl(JSONL_FILE)
dataset = MidiDataset(json_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)