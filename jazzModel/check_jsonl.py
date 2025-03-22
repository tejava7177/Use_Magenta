import gzip
import json

def preview_jsonl_gz(path, num_lines=50):
    lines = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            lines.append(json.loads(line))
    return lines

# ❗ 여기에 정확한 경로 입력!
jsonl_gz_path = "/Users/simjuheun/Desktop/개인프로젝트/Use_Magenta/jazzModel/jazz_dataset_transformer.jsonl.gz"
preview = preview_jsonl_gz(jsonl_gz_path)

for i, item in enumerate(preview[:5]):
    print(f"[{i+1}] {json.dumps(item, indent=2)}\n")