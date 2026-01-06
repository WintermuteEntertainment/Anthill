# prepare_datasets_parallel.py
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os

def extract_pairs(conversation):
    pairs = []
    msgs = conversation.get("messages", [])

    for i in range(len(msgs) - 1):
        a = msgs[i]
        b = msgs[i + 1]

        if a["role"] == "user" and b["role"] == "assistant":
            prompt = a["text"].strip()
            completion = b["text"].strip()

            if prompt and completion:
                pairs.append({
                    "prompt": prompt,
                    "completion": completion,
                    "conversation_id": conversation.get("id"),
                    "title": conversation.get("title")
                })

    return pairs


def main(input_path, output_path):
    # Use the arguments passed, not hardcoded paths!
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")

    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return 1

    conversations = data.get("conversations", [])
    print(f"Found {len(conversations)} conversations")
    
    if not conversations:
        print("No conversations found.")
        return 1

    # Count messages for debugging
    total_messages = sum(len(c.get("messages", [])) for c in conversations)
    print(f"Total messages: {total_messages}")

    # Use multiprocessing for speed
    print("Extracting prompt/response pairs...")
    with Pool(min(cpu_count(), 8)) as pool:
        results = pool.map(extract_pairs, conversations)

    total = 0
    try:
        with output_path.open("w", encoding="utf-8") as out:
            for group in results:
                for pair in group:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total += 1
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        return 1

    print(f"✅ Wrote {total} instruction pairs → {output_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: prepare_datasets_parallel.py <input_json> <output_jsonl>")
        print("Example: prepare_datasets_parallel.py chatgpt_conversations.json pairs.jsonl")
        sys.exit(1)

    sys.exit(main(sys.argv[1], sys.argv[2]))