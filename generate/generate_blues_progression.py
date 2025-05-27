import pickle

# 모델 로딩
model_path = '/Users/simjuheun/Desktop/myProject/Use_Magenta/train/train/blues_markov_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 사용자 입력
start = input("🎶 Starting chord (e.g., C, G7, Am): ").strip()

# 예외 처리 및 생성
if start not in model.transitions:
    print(f"⚠️ '{start}' not found in training data. Try a different chord from your dataset.")
else:
    generated = model.generate(start, length=16)
    print("\n🎸 Generated Blues Progression:")
    print(" → ".join(generated))