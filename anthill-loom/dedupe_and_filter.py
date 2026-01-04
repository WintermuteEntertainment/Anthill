import json
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])

seen = set()
kept = []

for line in inp.read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    key = (obj["prompt"], obj["completion"])
    if key in seen:
        continue
    if len(obj["prompt"]) < 5 or len(obj["completion"]) < 5:
        continue
    seen.add(key)
    kept.append(obj)

with out.open("w", encoding="utf-8") as f:
    for k in kept:
        f.write(json.dumps(k, ensure_ascii=False) + "\n")

print(f"Cleaned dataset: {len(kept)} pairs → {out}")
