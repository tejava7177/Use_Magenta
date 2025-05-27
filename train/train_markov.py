import json
import pickle
from model.markov_model import MarkovChordModel

# 로딩
data_path = './train/blues_chords.json'
with open(data_path) as f:
    sequences = json.load(f)

print(f"📁 Loaded {len(sequences)} chord sequences from {data_path}")

# 훈련
model = MarkovChordModel()
model.train(sequences)

# 저장
model_path = './train/blues_markov_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Blues Markov model trained and saved to: {model_path}")