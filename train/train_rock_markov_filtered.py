# train/train_rock_markov.py

import json
import pickle
from model.markov_model import MarkovChordModel

# 1. Load chord progression dataset
json_path = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/rock_chords_filtered.json'
print(f"📂 Loading chord data from: {json_path}")
with open(json_path) as f:
    sequences = json.load(f)

print(f"🎸 Loaded {len(sequences)} chord sequences.")

# 2. Train Markov model
print("🔁 Training MarkovChordModel...")
model = MarkovChordModel()
model.train(sequences)

# 3. Save trained model
output_path = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/rock_markov_model_filtered.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Rock Markov model trained and saved to {output_path}")