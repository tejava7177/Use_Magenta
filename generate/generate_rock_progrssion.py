# generate/generate_rock_progression.py

import pickle

# 1. Load trained Markov model
model_path = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/rock_markov_model_filtered.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 2. Get starting chord
start = input("🎶 Starting chord (e.g., C, G7, Am): ")

# 3. Generate progression
generated = model.generate(start, length=16)

# 4. Display result
print("\n🎸 Generated Rock Progression:")
print(" → ".join(generated))