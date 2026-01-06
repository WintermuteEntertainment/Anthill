# dedupe_and_filter.py
import json
import sys
from pathlib import Path

def main(input_path, output_path):
    inp = Path(input_path)
    out = Path(output_path)
    
    if not inp.exists():
        print(f"❌ Input file not found: {inp}")
        return 1
    
    seen = set()
    kept = []
    
    try:
        with inp.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    
                    # Extract prompt and completion
                    prompt = obj.get("prompt", "").strip()
                    completion = obj.get("completion", "").strip()
                    
                    # Skip if missing either
                    if not prompt or not completion:
                        continue
                    
                    # Create unique key (first 200 chars of each)
                    key = (prompt[:200], completion[:200])
                    
                    # Skip duplicates
                    if key in seen:
                        continue
                    
                    # Skip very short pairs
                    if len(prompt) < 5 or len(completion) < 5:
                        continue
                    
                    # Skip very long pairs (likely errors)
                    if len(prompt) > 10000 or len(completion) > 10000:
                        continue
                    
                    # Keep this pair
                    seen.add(key)
                    kept.append({
                        "prompt": prompt,
                        "completion": completion,
                        # Keep metadata if present
                        "conversation_id": obj.get("conversation_id"),
                        "title": obj.get("title")
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} invalid JSON: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return 1
    
    # Save cleaned data
    try:
        with out.open("w", encoding="utf-8") as f:
            for k in kept:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        return 1
    
    print(f"✅ Cleaned dataset: {len(kept)} pairs → {out}")
    print(f"   (Removed {len(seen) - len(kept)} duplicates/short pairs)")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: dedupe_and_filter.py <input_jsonl> <output_jsonl>")
        print("Example: dedupe_and_filter.py pairs.jsonl pairs_clean.jsonl")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1], sys.argv[2]))