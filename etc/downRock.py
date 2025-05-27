# etc/downRock.py

import kagglehub
import pandas as pd
import json
import os

# 1. Download dataset
print("📦 Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("henryshan/a-dataset-of-666000-chord-progressions")
print(f"✅ Dataset downloaded to: {path}")

# 2. Detect CSV file
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in dataset folder.")
csv_name = csv_files[0]
print(f"📄 Found CSV file: {csv_name}")

# 3. Load CSV
print("📂 Reading CSV file...")
df = pd.read_csv(os.path.join(path, csv_name), low_memory=False)
print(f"🧾 Total songs in dataset: {len(df)}")
print("🧾 Available columns in dataset:")
print(df.columns.tolist())

# 4. Filter Rock genre using 'main_genre'
print("🎸 Filtering Rock genre songs using 'main_genre' column...")
rock_df = df[df["main_genre"].str.lower().str.contains("rock", na=False)]
print(f"✅ Total Rock songs found: {len(rock_df)}")

# 5. Convert chord strings to list of sequences
print("🎼 Converting chord progressions to sequences...")
chord_sequences = rock_df["chords"].dropna().str.split(" ")
chord_list = chord_sequences.tolist()

# Preview
print("\n🔍 Sample Rock chord progressions:")
for i in range(min(3, len(chord_list))):
    print(f"   {i+1}: {' → '.join(chord_list[i][:10])} ...")

# 6. Save to JSON
print("\n💾 Saving to train/rock_chords.json...")
os.makedirs("./train", exist_ok=True)
with open("../train/train/rock_chords.json", "w") as f:
    json.dump(chord_list, f)

print("🎯 Done! Rock chord sequences saved to ./train/rock_chords.json")