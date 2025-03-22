import gzip
import json

def preview_jsonl_gz(path, num_lines=5):
    print(f"🔍 파일 미리보기 (최대 {num_lines}줄): {path}\n")
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                item = json.loads(line)
                print(f"[{i + 1}] 파일: {item.get('filename')}")
                print(f"  - 트랙 이름: {item.get('track_name')}")
                print(f"  - 이벤트 수: {len(item.get('events', []))}")
                print(f"  - 예시 이벤트: {item.get('events')[:3]}")
                print()
    except Exception as e:
        print(f"❌ 파일을 여는 중 오류 발생: {e}")

# ✅ 경로 입력 (수정 필요할 수도 있음)
preview_jsonl_gz("jazz_piano_dataset.jsonl.gz", num_lines=5)