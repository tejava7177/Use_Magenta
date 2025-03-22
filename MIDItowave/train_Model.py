import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# -------------------------------
# Custom Dataset
# -------------------------------

class JazzMelDataset(Dataset):
    def __init__(self, jsonl_path):
        self.entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        features = item['features']
        x = np.array([
            features['tempo'],
            features['duration'],
            features['pitch_mean'],
            features['pitch_min'],
            features['pitch_max'],
            features['rms_energy'],
            features['onset_density'],
            *features['mfcc_mean']
        ], dtype=np.float32)

        mel = np.load(item['mel_path'])  # shape: (80, T)
        mel = mel[:, :400] if mel.shape[1] > 400 else np.pad(mel, ((0, 0), (0, 400 - mel.shape[1])))
        return x, mel


# -------------------------------
# MLP Model
# -------------------------------

class ConditionToMelNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=80 * 400):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Training Loop
# -------------------------------

def train_model(jsonl_path, epochs=10, batch_size=8, save_path="mel_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JazzMelDataset(jsonl_path)

    input_dim = len(dataset[0][0])
    output_dim = dataset[0][1].shape[0] * dataset[0][1].shape[1]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ConditionToMelNet(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.view(y.size(0), -1).to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.view(y.size(0), -1).to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

        print(
            f"📚 Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")
    return model