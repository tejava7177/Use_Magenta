from train_Model import train_model

model = train_model(
    "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/MIDItowave/mini_dataset_100.jsonl",
    epochs=10,
    batch_size=8
)