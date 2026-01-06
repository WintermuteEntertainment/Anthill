import json
import sys
from pathlib import Path

inp = Path(r"X:\Anthill\Anthill\anthill-loom\datasets\processed\pairs_clean.jsonl")
out = Path(r"X:\Anthill\Anthill\anthill-loom\datasets\processed\pairs_clean_and_extracted.jsonl")

data = json.loads(inp.read_text(encoding="utf-8"))

pairs = []
for conv in data.get("conversations", []):
    msgs = conv.get("messages", [])
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
            pairs.append({
                "prompt": msgs[i]["text"].strip(),
                "completion": msgs[i+1]["text"].strip()
            })

with out.open("w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Wrote {len(pairs)} instruction pairs → {out}")
