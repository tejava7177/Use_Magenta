# train/prepare_filtered_rock_data.py

import json
import re

INPUT_PATH = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/rock_chords.json'
OUTPUT_PATH = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/rock_chords_filtered.json'

with open(INPUT_PATH) as f:
    data = json.load(f)

# 마커 패턴: <...>
def is_valid_chord(chord):
    return not re.match(r"^<.*?>$", chord)

# 마커 제거
filtered_data = []
for seq in data:
    cleaned = [chord for chord in seq if is_valid_chord(chord)]
    if len(cleaned) > 2:
        filtered_data.append(cleaned)

# 저장
with open(OUTPUT_PATH, 'w') as f:
    json.dump(filtered_data, f)

print(f"✅ Filtered chord sequences saved to: {OUTPUT_PATH}")
print(f"🧼 Total sequences after cleaning: {len(filtered_data)}")