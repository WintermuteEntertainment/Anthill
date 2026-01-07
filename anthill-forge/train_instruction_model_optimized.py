# train_instruction_model_optimized.py - WSL/Windows Hybrid
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
import time
import os
import platform

# -----------------------------
# AUTO-CONFIG BASED ON OS
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Detect WSL vs Windows
IS_WSL = 'WSL' in platform.uname().release or 'microsoft' in platform.uname().release.lower()
IS_WINDOWS = platform.system() == 'Windows'

print(f"🚀 Platform: {'WSL' if IS_WSL else 'Windows'}")

# Auto-configure paths
if IS_WSL:
    # WSL paths (Linux style)
    MODEL_PATH = Path("/mnt/z/gertrude_phi2_finetune/final_model")
    DATA_FOLDER = Path("/mnt/x/Anthill/Anthill/anthill-loom/datasets/processed")
    OUTPUT_DIR = Path("/mnt/c/anthill_forge_output")
    
    # WSL can use multiprocessing
    DATALOADER_WORKERS = 4
    DATALOADER_PIN_MEMORY = True
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
else:
    # Windows paths (UNC style)
    MODEL_PATH = Path("//BigBlackBox/a/gertrude_phi2_finetune/final_model")
    DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
    OUTPUT_DIR = Path("C:/anthill_forge_output")
    
    # Windows limited multiprocessing
    DATALOADER_WORKERS = 0  # 0 or 1 on Windows
    DATALOADER_PIN_MEMORY = False
    
# Training hyperparameters
MAX_LENGTH = 1024
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-5

# -----------------------------
# OPTIMIZED MODEL LOADING
# -----------------------------
def load_model_optimized(device="cuda"):
    """Optimized model loading with memory monitoring"""
    print(f"\n🤖 Loading model from: {MODEL_PATH}")
    start_time = time.time()
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    
    if device == "cuda":
        # Use optimal precision
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("   Using bfloat16 (optimal)")
        else:
            dtype = torch.float16
            print("   Using float16")
        
        load_kwargs.update({
            "torch_dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        })
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            **load_kwargs
        )
        
        # Enable gradient checkpointing BEFORE any forward pass
        model.gradient_checkpointing_enable()
        
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Model dtype: {model.dtype}")
        print(f"   Gradient checkpointing: Enabled")
        
        # Monitor initial memory
        self.monitor_memory("After model load")
        
        return model
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL: {e}")
        raise

# -----------------------------
# MEMORY MONITORING CLASS
# -----------------------------
class MemoryMonitor:
    """Monitor and optimize memory during training"""
    def __init__(self):
        self.peak_memory = 0
        
    def monitor_memory(self, stage=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.peak_memory = max(self.peak_memory, allocated)
            
            print(f"   Memory {stage}: {allocated:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved (Peak: {self.peak_memory:.2f}GB)")
    
    def cleanup(self):
        """Force garbage collection"""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("   Memory cleanup completed")

# -----------------------------
# OPTIMIZED TRAINING ARGUMENTS
# -----------------------------
def get_training_args(output_dir, use_bf16, total_steps):
    """Get optimized training arguments based on platform"""
    
    # Platform-specific optimizations
    if IS_WSL:
        # WSL can handle more workers
        num_workers = DATALOADER_WORKERS
        pin_memory = True
    else:
        # Windows: minimal multiprocessing
        num_workers = 0
        pin_memory = False
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=min(100, total_steps // 10),
        
        # Save strategy - optimize for network training
        save_strategy="steps",
        save_steps=total_steps // 3,  # Save 3 times total
        save_total_limit=1,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=total_steps // 5,  # Evaluate 5 times total
        load_best_model_at_end=False,
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        report_to="none",
        
        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,
        fp16_full_eval=False,
        
        # Performance optimizations
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        
        # Data loading (platform-specific)
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        
        # Memory optimizations
        remove_unused_columns=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,  # Avoid partial batches
        
        # Network training optimizations
        push_to_hub=False,
        resume_from_checkpoint=None,
        ignore_data_skip=False,
        
        # Disable unnecessary metrics
        metric_for_best_model=None,
        greater_is_better=None,
    )
    
    return args

# -----------------------------
# MAIN OPTIMIZED FUNCTION
# -----------------------------
def main():
    print("=" * 70)
    print("ANTHILL FORGE - OPTIMIZED FOR WSL/WINDOWS")
    print("=" * 70)
    print(f"Platform: {'WSL (Linux)' if IS_WSL else 'Windows'}")
    print(f"Data workers: {DATALOADER_WORKERS}")
    print(f"Pin memory: {DATALOADER_PIN_MEMORY}")
    print("=" * 70)
    
    # Initialize memory monitor
    mem_monitor = MemoryMonitor()
    
    # Step 1: Verify paths
    print(f"\n📁 Model path: {MODEL_PATH}")
    print(f"📁 Data path: {DATA_FOLDER}")
    print(f"📁 Output path: {OUTPUT_DIR}")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Load tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 3: Load and prepare data
    print(f"\n📊 Loading datasets...")
    files = list(DATA_FOLDER.rglob("*_clean.jsonl"))
    
    if not files:
        print(f"❌ No cleaned datasets found")
        return 1
    
    print(f"Found {len(files)} dataset(s)")
    
    # Load and merge (with progress)
    datasets = []
    for f in files:
        print(f"  Loading {f.name}...")
        try:
            ds = load_dataset("json", data_files=str(f))["train"]
            datasets.append(ds)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if not datasets:
        print("❌ No data loaded")
        return 1
    
    dataset = concatenate_datasets(datasets)
    print(f"✅ Total examples: {len(dataset):,}")
    
    # Step 4: Format and tokenize
    def format_example(ex):
        return {"text": f"Human: {ex['prompt'].strip()}\n\nAssistant: {ex['completion'].strip()}"}
    
    print("Formatting dataset...")
    formatted = dataset.map(format_example, remove_columns=dataset.column_names)
    
    print("Tokenizing dataset...")
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])
    
    # Split
    split_data = tokenized.train_test_split(test_size=0.1, seed=42)
    train_data = split_data["train"]
    eval_data = split_data["test"]
    
    print(f"Train: {len(train_data):,}, Eval: {len(eval_data):,}")
    
    # Step 5: Calculate steps
    effective_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = max(1, len(train_data) // effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print(f"\n📈 Training stats:")
    print(f"   Effective batch size: {effective_batch}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    
    # Step 6: Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_bf16_supported()
    
    model = load_model_optimized(device)
    mem_monitor.monitor_memory("After model load")
    
    # Step 7: Setup training
    training_args = get_training_args(OUTPUT_DIR, use_bf16, total_steps)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Step 8: Train with monitoring
    print(f"\n🚀 Starting training...")
    print(f"   Total steps: {total_steps}")
    print(f"   Estimated time: {total_steps * 45 / 3600:.1f} hours")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Train with periodic memory cleanup
        for epoch in range(NUM_EPOCHS):
            print(f"\n📚 Epoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train this epoch
            trainer.train()
            
            # Cleanup between epochs
            mem_monitor.cleanup()
            mem_monitor.monitor_memory(f"After epoch {epoch+1}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupted = True
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    else:
        interrupted = False
    
    # Step 9: Save final model
    print(f"\n💾 Saving model...")
    save_dir = OUTPUT_DIR / ("interrupted" if interrupted else "final")
    save_dir.mkdir(exist_ok=True)
    
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    
    # Save metadata
    metadata = {
        "platform": "WSL" if IS_WSL else "Windows",
        "model_path": str(MODEL_PATH),
        "output_path": str(save_dir),
        "dataset_size": len(dataset),
        "training_examples": len(train_data),
        "total_steps": total_steps,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "precision": "bfloat16" if use_bf16 else "float16",
        "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "interrupted": interrupted,
        "peak_gpu_memory_gb": mem_monitor.peak_memory,
        "dataloader_workers": DATALOADER_WORKERS,
    }
    
    with open(save_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training {'interrupted' if interrupted else 'completed'}!")
    print(f"   Model saved to: {save_dir}")
    print(f"   Peak GPU memory: {mem_monitor.peak_memory:.2f}GB")
    print(f"   Platform: {'WSL' if IS_WSL else 'Windows'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
