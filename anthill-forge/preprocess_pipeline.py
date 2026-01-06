# preprocess_pipeline.py
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: preprocess_pipeline.py <input_json> <output_dir>")
        print("Example: preprocess_pipeline.py chatgpt_conversations_latest.json datasets/processed")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract pairs
    print("Step 1: Extracting prompt/completion pairs...")
    pairs_path = output_dir / "pairs.jsonl"
    
    # You can either import and run prepare_datasets_parallel.py logic here
    # Or call it as a subprocess
    import subprocess
    subprocess.run([
        sys.executable, "prepare_datasets_parallel.py",
        str(input_path), str(pairs_path)
    ])
    
    # Step 2: Deduplicate and clean
    print("Step 2: Deduplicating and filtering...")
    clean_path = output_dir / "pairs_clean.jsonl"
    
    subprocess.run([
        sys.executable, "dedupe_and_filter.py",
        str(pairs_path), str(clean_path)
    ])
    
    # Report
    with open(pairs_path, 'r', encoding='utf-8') as f:
        raw_count = sum(1 for _ in f)
    
    with open(clean_path, 'r', encoding='utf-8') as f:
        clean_count = sum(1 for _ in f)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Raw pairs: {raw_count}")
    print(f"   Clean pairs: {clean_count}")
    print(f"   Removed: {raw_count - clean_count} duplicates/short pairs")
    print(f"   Clean file: {clean_path}")
    
    return clean_path

if __name__ == "__main__":
    main()
