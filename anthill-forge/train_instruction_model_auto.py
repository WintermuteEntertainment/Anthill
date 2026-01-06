# train_instruction_model_auto.py - GPU Optimized
import torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
import sys
import os

# -----------------------------
# CONFIG
# -----------------------------
# Use relative paths from this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Model path - adjust this to your actual model location
MODEL_NAME = "microsoft/phi-2"  # or your custom model path

# Data paths
DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "anthill_forge_output"
MAX_LENGTH = 2048

# Training hyperparameters - OPTIMIZED FOR GPU
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4  # Increased for GPU (4070 has 12GB VRAM)
GRAD_ACCUM_STEPS = 8       # Effective batch size = 32
LEARNING_RATE = 2e-5

# -----------------------------
# GPU CHECK & OPTIMIZATION
# -----------------------------
def setup_gpu():
    """Check and setup GPU with optimal settings"""
    print("\n🔍 Checking GPU availability...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA is available! Found {gpu_count} GPU(s):")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Total VRAM: {memory_total:.1f} GB")
            print(f"    Allocated: {memory_allocated:.2f} GB")
            print(f"    Reserved: {memory_reserved:.2f} GB")
            
            # Set this GPU as default
            if i == 0:
                torch.cuda.set_device(i)
                print(f"    ✓ Using GPU {i} for training")
        
        return "cuda"
    else:
        print("❌ CUDA not available. Check:")
        print("   1. PyTorch installed with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   2. NVIDIA drivers are up to date")
        print("   3. CUDA toolkit is installed")
        return "cpu"

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def get_all_jsonl_files(folder: Path):
    """Recursively find all JSONL files in a folder"""
    return list(folder.rglob("*_clean.jsonl"))

def merge_datasets(files):
    """Load multiple JSONL files and concatenate them into a single dataset"""
    datasets_list = []
    for f in files:
        print(f"  Loading {f.name}...")
        try:
            ds = load_dataset("json", data_files=str(f))["train"]
            datasets_list.append(ds)
        except Exception as e:
            print(f"  ❌ Error loading {f}: {e}")
    return concatenate_datasets(datasets_list)

def format_for_training(example):
    """Turn prompt/completion pair into text for instruction tuning"""
    prompt = example['prompt'].strip()
    completion = example['completion'].strip()
    
    # Format that works well with phi-2
    formatted_text = f"Instruct: {prompt}\nOutput: {completion}"
    
    return {"text": formatted_text}

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 60)
    print("Anthill Forge - Instruction Tuning Pipeline")
    print("=" * 60)
    
    # Step 0: Setup GPU
    device = setup_gpu()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find cleaned datasets
    print(f"\n📊 Step 1: Looking for cleaned datasets in {DATA_FOLDER}")
    
    if not DATA_FOLDER.exists():
        print(f"❌ Data folder not found: {DATA_FOLDER}")
        print("\nPlease run preprocessing first:")
        print("1. Use Anthill Spider extension to download conversations")
        print("2. Run anthill-loom\\run_chatgpt_pipeline.bat with the JSON file")
        return 1
    
    files = get_all_jsonl_files(DATA_FOLDER)
    if not files:
        print(f"❌ No *_clean.jsonl files found in {DATA_FOLDER}")
        print("\nMake sure you've run the preprocessing pipeline.")
        return 1
    
    print(f"Found {len(files)} cleaned dataset(s):")
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            line_count = sum(1 for _ in file)
        print(f"  • {f.name}: {line_count:,} pairs")
    
    # Step 2: Load and merge datasets
    print("\n📦 Step 2: Loading and merging datasets...")
    dataset = merge_datasets(files)
    
    if len(dataset) == 0:
        print("❌ No data loaded. Check your cleaned JSONL files.")
        return 1
    
    print(f"Total training examples: {len(dataset):,}")
    
    # Step 3: Format for training
    print("\n🔄 Step 3: Formatting examples for instruction tuning...")
    formatted_dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)
    
    # Step 4: Load tokenizer
    print("\n🔤 Step 4: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return 1
    
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Step 5: Split into train/validation
    print("Splitting dataset (90% train, 10% validation)...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(eval_dataset):,}")
    
    # Step 6: Load model with GPU optimization
    print(f"\n🤖 Step 5: Loading model for {device.upper()}...")
    
    try:
        if device == "cuda":
            # Optimize for GPU
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Add this
            )
            
            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
        else:
            # CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTrying alternative loading method...")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            if device == "cuda":
                model = model.to("cuda")
        except Exception as e2:
            print(f"❌ Alternative loading also failed: {e2}")
            return 1
    
    # Step 7: Training arguments optimized for GPU
    print("\n⚙️ Step 6: Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",  # Mixed precision for GPU
        bf16=False,  # Set to True if using Ampere GPU (RTX 30/40 series)
        gradient_checkpointing=True,  # Save memory
        optim="adamw_torch_fused",  # Faster optimizer
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4 if device == "cuda" else 0,  # More workers for GPU
        remove_unused_columns=False,
    )
    
    # Step 8: Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Step 9: Train with GPU monitoring
    print(f"\n🚀 Step 7: Starting training on {device.upper()}...")
    print(f"Training on {len(train_dataset):,} examples")
    print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Total steps: ~{len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS) * NUM_EPOCHS:,}")
    
    if device == "cuda":
        print(f"\n📊 GPU Memory Before Training:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("\n" + "-" * 60)
    
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU Out of Memory! Try:")
        print("   1. Reduce PER_DEVICE_BATCH_SIZE to 2")
        print("   2. Increase GRAD_ACCUM_STEPS to 16")
        print("   3. Reduce MAX_LENGTH to 1024")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return 1
    
    # Step 10: Save model
    print("\n💾 Step 8: Saving model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Save training metadata
    metadata = {
        "model": MODEL_NAME,
        "device": device,
        "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
        "dataset_files": [str(f) for f in files],
        "total_examples": len(dataset),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    }
    
    with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    # Show final metrics
    if trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        if "loss" in final_log:
            print(f"   Final training loss: {final_log['loss']:.4f}")
        if "eval_loss" in final_log:
            print(f"   Final validation loss: {final_log['eval_loss']:.4f}")
    
    if device == "cuda":
        print(f"\n📊 GPU Memory After Training:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print(f"\n🎉 Anthill Forge pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())