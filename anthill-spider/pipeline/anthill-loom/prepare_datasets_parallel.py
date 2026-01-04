# prepare_datasets_parallel.py

import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

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
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    if not conversations:
        print("No conversations found.")
        return

    with Pool(cpu_count()) as pool:
        results = pool.map(extract_pairs, conversations)

    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for group in results:
            for pair in group:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} instruction pairs → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: prepare_datasets_parallel.py <input_json> <output_jsonl>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
