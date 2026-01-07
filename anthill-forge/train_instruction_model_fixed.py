# train_instruction_model_strict.py - FIXED FP16 ISSUE
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

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# STRICT: Must load from this exact path
MODEL_PATH = Path("//BigBlackBox/a/gertrude_phi2_finetune/final_model")
DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"

# CRITICAL: Change output to LOCAL DRIVE (C: or D:, NOT X:)
OUTPUT_DIR = Path("C:/anthill_forge_output")  # <-- LOCAL SSD

MAX_LENGTH = 1024  # Reduced from 2048 for memory efficiency

# Adjusted hyperparameters for 4070 12GB
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1  # Reduced to 1 to be safe
GRAD_ACCUM_STEPS = 16      # Increased to compensate
LEARNING_RATE = 2e-5

# -----------------------------
# STRICT VERIFICATION FUNCTIONS
# -----------------------------
def verify_model_directory():
    """Verify the model directory exists and has required files"""
    print(f"\n🔍 Verifying model directory: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"❌ CRITICAL: Model directory does not exist!")
        print(f"   Path: {MODEL_PATH}")
        print(f"   Absolute path: {MODEL_PATH.absolute()}")
        return False
    
    print(f"✅ Directory exists")
    
    # Quick check - just see if model files exist
    model_files = list(MODEL_PATH.glob("model*.safetensors"))
    if not model_files:
        print(f"❌ No model.safetensors files found")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    return True

def load_tokenizer_strict():
    """Load tokenizer strictly from the specified path"""
    print(f"\n🔤 Loading tokenizer from: {MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Set padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Model max length: {tokenizer.model_max_length}")
        
        return tokenizer
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD TOKENIZER: {e}")
        raise

def load_model_strict(device="cuda"):
    """Load model strictly from the specified path"""
    print(f"\n🤖 Loading model from: {MODEL_PATH}")
    start_time = time.time()
    
    # Build loading arguments - Use bfloat16 instead of float16
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    
    if device == "cuda":
        print("   Using CUDA with bfloat16 precision...")
        # Use bfloat16 instead of float16 to avoid gradient scaling issues
        if torch.cuda.is_bf16_supported():
            load_kwargs.update({
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            })
        else:
            # Fallback to float16
            load_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
            })
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), **load_kwargs)
        load_time = time.time() - start_time
        
        if device == "cuda":
            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            
            print(f"✅ Model loaded in {load_time:.1f} seconds")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Model dtype: {model.dtype}")
            
            # Show memory usage
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"   Memory allocated: {allocated:.2f} GB")
            print(f"   Memory reserved: {reserved:.2f} GB")
        
        return model
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL: {e}")
        raise

# -----------------------------
# DATA FUNCTIONS
# -----------------------------
def get_all_jsonl_files(folder: Path):
    return list(folder.rglob("*_clean.jsonl"))

def merge_datasets(files):
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
    prompt = example['prompt'].strip()
    completion = example['completion'].strip()
    formatted_text = f"Human: {prompt}\n\nAssistant: {completion}"
    return {"text": formatted_text}

# -----------------------------
# MAIN TRAINING FUNCTION
# -----------------------------
def main():
    print("=" * 70)
    print("ANTHILL FORGE - FP16 FIX EDITION")
    print("=" * 70)
    print("Fixed: FP16 gradient scaling issue")
    print("Using bfloat16 if available, else float16 with proper settings")
    print("=" * 70)
    
    # Step 0: Quick model verification
    print("\n🚨 STEP 0: MODEL VERIFICATION")
    print("-" * 40)
    if not verify_model_directory():
        print("\n❌ CANNOT CONTINUE - Model verification failed")
        return 1
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Hardware: {device.upper()}")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
        
        # Check bfloat16 support
        if torch.cuda.is_bf16_supported():
            print("   ✅ GPU supports bfloat16")
        else:
            print("   ⚠️  GPU does NOT support bfloat16, using float16")
    
    # Verify output directory
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get user confirmation
    print(f"\n📊 Will train using model from: {MODEL_PATH}")
    response = input("\n❓ Continue with training? (yes/NO): ").strip().lower()
    if response != 'yes':
        print("\n⏹️  Training cancelled.")
        return 0
    
    # Step 1: Find cleaned datasets
    print(f"\n📊 STEP 1: LOADING DATASETS")
    print("-" * 40)
    
    if not DATA_FOLDER.exists():
        print(f"❌ Data folder not found: {DATA_FOLDER}")
        return 1
    
    files = get_all_jsonl_files(DATA_FOLDER)
    if not files:
        print(f"❌ No *_clean.jsonl files found")
        return 1
    
    print(f"Found {len(files)} cleaned dataset(s):")
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            line_count = sum(1 for _ in file)
        print(f"   • {f.name}: {line_count:,} pairs")
    
    # Step 2: Load and merge datasets
    print("\n📦 Loading and merging datasets...")
    dataset = merge_datasets(files)
    
    if len(dataset) == 0:
        print("❌ No data loaded.")
        return 1
    
    print(f"✅ Total training examples: {len(dataset):,}")
    
    # Step 3: Format for training
    print("\n🔄 Formatting examples for instruction tuning...")
    formatted_dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)
    
    # Step 4: Load tokenizer
    print(f"\n🔤 STEP 2: LOADING TOKENIZER")
    print("-" * 40)
    try:
        tokenizer = load_tokenizer_strict()
    except Exception as e:
        return 1
    
    # Tokenize dataset with smaller max_length
    print("\n🔡 Tokenizing dataset...")
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=MAX_LENGTH,  # Reduced to 1024
            padding="max_length"
        )
    
    tokenized_dataset = formatted_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Split dataset
    print("Splitting dataset (90% train, 10% validation)...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"   Training examples: {len(train_dataset):,}")
    print(f"   Validation examples: {len(eval_dataset):,}")
    
    # Step 5: Load model
    print(f"\n🤖 STEP 3: LOADING MODEL")
    print("-" * 40)
    try:
        model = load_model_strict(device=device)
    except Exception as e:
        return 1
    
    # Calculate training steps
    effective_batch_size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print(f"\n📈 Training stats:")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    
    # Step 6: Configure training - SIMPLIFIED
    print(f"\n⚙️ STEP 4: CONFIGURING TRAINING")
    print("-" * 40)
    
    # Determine precision settings based on GPU support
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16  # Only use fp16 if bf16 is not supported
    
    print(f"   Using {'bfloat16' if use_bf16 else 'float16'} precision")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        
        # Save settings - ONLY at end
        save_strategy="no",  # Don't save during training
        save_steps=999999,   # Very high number
        save_total_limit=0,  # Don't save checkpoints
        
        # Evaluation - minimal
        eval_strategy="no",  # Disable evaluation during training
        eval_steps=999999,
        load_best_model_at_end=False,
        
        # Logging
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        
        # Precision settings - FIXED
        bf16=use_bf16,      # Use bfloat16 if supported
        fp16=use_fp16,      # Use float16 only if bf16 not supported
        fp16_full_eval=False,  # Don't use fp16 for evaluation
        
        # Performance
        gradient_checkpointing=True,
        optim="adamw_torch",
        
        # Disable gradient clipping to avoid scaling issues
        max_grad_norm=1.0,
        
        # Data loading
        dataloader_pin_memory=False,  # Can cause issues with large datasets
        dataloader_num_workers=0,     # Disable multiprocessing
        
        # Disable unnecessary features
        remove_unused_columns=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
    )
    
    # Step 7: Create trainer
    print("\n👨‍🏫 Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Keep for final evaluation
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Step 8: Train!
    print(f"\n🚀 STEP 5: TRAINING")
    print("=" * 70)
    print(f"Starting training on {len(train_dataset):,} examples")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total steps: ~{total_steps}")
    print(f"Expected time: ~{total_steps * 3 / 3600:.1f} hours (at 3 sec/step)")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 70)
    
    try:
        # Start training with progress callback
        print("\n▶️  Training started...")
        
        # # Add a simple callback to show progress
        # class ProgressCallback:
        #     def on_log(self, args, state, control, logs=None, **kwargs):
        #         if logs and 'loss' in logs:
        #             print(f"Step {state.global_step}/{total_steps} - Loss: {logs['loss']:.4f}")
        
        # trainer.add_callback(ProgressCallback())
        #using default logging instead
        
        # Train!
        trainer.train()
        
        print(f"\n✅ Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU OUT OF MEMORY!")
        print("   Try reducing PER_DEVICE_BATCH_SIZE to 1")
        return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ TRAINING INTERRUPTED BY USER")
        print("   Will save current state...")
        # Continue to save step below
        interrupted = True
        
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    else:
        interrupted = False
    
    # Step 9: Save final model
    print(f"\n💾 SAVING FINAL MODEL")
    print("-" * 40)
    
    try:
        if interrupted:
            save_dir = OUTPUT_DIR / "interrupted"
            print(f"   Saving interrupted training to: {save_dir}")
        else:
            save_dir = OUTPUT_DIR
            print(f"   Saving final model to: {save_dir}")
        
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        
        print(f"✅ Model saved successfully")
        
        # Show saved files
        print(f"\n📁 Saved files:")
        for item in save_dir.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   • {item.name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
    
    # Save metadata
    metadata = {
        "source_model": str(MODEL_PATH),
        "training_device": device,
        "dataset_size": len(dataset),
        "training_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "epochs_completed": NUM_EPOCHS,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "precision": "bfloat16" if use_bf16 else "float16",
        "completion_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "interrupted": interrupted,
    }
    
    try:
        with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✅ Metadata saved")
    except Exception as e:
        print(f"⚠️  Could not save metadata: {e}")
    
    # Final report
    print(f"\n{'⚠️' if interrupted else '🎉'} {'TRAINING INTERRUPTED' if interrupted else 'TRAINING COMPLETE'}!")
    print("=" * 40)
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    if trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        if "loss" in final_log:
            print(f"   Final training loss: {final_log['loss']:.4f}")
    
    print(f"\n   Next steps:")
    print(f"   1. Test the model locally")
    print(f"   2. Copy to network when ready")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
