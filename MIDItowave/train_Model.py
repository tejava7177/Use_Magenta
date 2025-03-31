import os
import json
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


# ✅ AttrDict for config if needed
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# ✅ Dataset Class
class FastMelDataset(Dataset):
    def __init__(self, jsonl_path, max_frames=100):
        self.entries = []
        self.max_frames = max_frames
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if os.path.exists(item['mel_path']) and os.path.exists(item['audio_path']):
                    self.entries.append(item)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        features = item['features']

        x = np.array([
            features['tempo'], features['duration'],
            features['pitch_mean'], features['pitch_min'], features['pitch_max'],
            features['rms_energy'], features['onset_density'],
            *features['mfcc_mean']
        ], dtype=np.float32)

        mel = np.load(item['mel_path'])  # (80, T)
        T = mel.shape[1]
        if T >= self.max_frames:
            mel = mel[:, :self.max_frames]
        else:
            pad_width = self.max_frames - T
            mel = np.pad(mel, ((0, 0), (0, pad_width)))

        return torch.FloatTensor(x), torch.FloatTensor(mel)


# ✅ CNN-based Mel Decoder
class MelDecoder(nn.Module):
    def __init__(self, input_dim=20, mel_bins=80, frames=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, mel_bins * frames)
        )
        self.mel_bins = mel_bins
        self.frames = frames

    def forward(self, x):
        out = self.fc(x)  # (B, mel_bins * frames)
        return out.view(-1, self.mel_bins, self.frames)  # (B, 80, 100)


# ✅ Training Loop

def train_mel_generator(jsonl_path, epochs=10, batch_size=32, save_path="mel_generator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FastMelDataset(jsonl_path)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = MelDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        print(f"📚 Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    train_mel_generator(
        jsonl_path="/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/mini_dataset_3200.jsonl",
        epochs=15,
        batch_size=32,
        save_path="mel_generator_fast.pth"
    )