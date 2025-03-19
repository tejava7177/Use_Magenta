# midi_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from midi_preprocessing import dataset, dataloader  # ✅ 데이터셋 & DataLoader 가져오기

# ✅ Transformer 모델 정의
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

# ✅ 모델 설정
DEVICE = torch.device("cpu")
model = MidiTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 함수
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output.view(-1, 4), batch.view(-1, 4))  # ✅ 손실 계산 변경
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()  # ✅ detach() 사용하여 그래프 분리

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")

# ✅ 학습 실행
train(model, dataloader, criterion, optimizer)
print("🎶 Transformer 모델 학습 완료! 🚀")