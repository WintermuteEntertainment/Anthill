# train_instruction_model_auto.py - Optimized for network loading and timeout protection
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
import signal
import threading
import time
import psutil
import os

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Network model path (use UNC path for Windows)
MODEL_NAME = r"Z:/gertrude_phi2_finetune/final_model"
# Alternative: Map network drive first then use local path
# MODEL_NAME = "Z:/gertrude_phi2_finetune/final_model"  # After mapping

DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "anthill_forge_output"
MAX_LENGTH = 2048

# Training hyperparameters
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2  # Reduced for network loading
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5

# Timeout and monitoring
MAX_TRAINING_HOURS = 12  # Maximum training time
HEARTBEAT_INTERVAL = 300  # 5 minutes
CHECKPOINT_INTERVAL = 1800  # 30 minutes

# -----------------------------
# TIMEOUT AND MONITORING
# -----------------------------
class TimeoutMonitor:
    """Monitors training for hangs and timeouts"""
    
    def __init__(self, max_hours=12):
        self.max_seconds = max_hours * 3600
        self.start_time = None
        self.last_progress_time = None
        self.is_running = False
        self.heartbeat_thread = None
        self.timeout_event = threading.Event()
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        self.is_running = True
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()
        
        print(f"⏱️  Timeout monitor started (max: {self.max_seconds/3600:.1f} hours)")
        
    def _heartbeat_monitor(self):
        """Monitor for hangs"""
        while self.is_running and not self.timeout_event.is_set():
            current_time = time.time()
            elapsed = current_time - self.last_progress_time
            
            # Check for hang (no progress in 30 minutes)
            if elapsed > 1800:  # 30 minutes
                print(f"⚠️  WARNING: No progress for {elapsed/60:.1f} minutes!")
                # Try to recover by checking GPU
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated(0) / 1e9
                        print(f"GPU memory allocated: {allocated:.2f} GB")
                        if allocated < 0.1:
                            print("❌ GPU appears idle - possible hang!")
                            self.timeout_event.set()
                    except:
                        pass
            
            # Check total time
            if current_time - self.start_time > self.max_seconds:
                print(f"⏰ Maximum training time reached ({self.max_seconds/3600:.1f} hours)")
                self.timeout_event.set()
                break
                
            time.sleep(HEARTBEAT_INTERVAL)
    
    def update_progress(self):
        """Update last progress time"""
        self.last_progress_time = time.time()
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
    
    def should_stop(self):
        """Check if training should stop"""
        return self.timeout_event.is_set()

def setup_timeout_handler():
    """Setup signal handlers for graceful termination"""
    def signal_handler(sig, frame):
        print(f"\n⚠️  Received signal {sig}, initiating graceful shutdown...")
        raise KeyboardInterrupt()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("✅ Signal handlers installed for graceful shutdown")

# -----------------------------
# NETWORK MODEL LOADING
# -----------------------------
def load_model_over_network(model_path, device="cuda"):
    """Load model from network path with optimizations"""
    print(f"🌐 Loading model from network path: {model_path}")
    
    # Options for network loading
    load_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
    }
    
    if device == "cuda":
        load_kwargs.update({
            "device_map": "auto",
            "offload_folder": str(OUTPUT_DIR / "offload"),  # Offload to disk if needed
        })
        
        # Enable gradient checkpointing to save memory
        load_kwargs["use_cache"] = False
    
    # For network paths, add retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Loading model (attempt {attempt + 1}/{max_retries})...")
            
            # Create offload directory if needed
            if "offload_folder" in load_kwargs:
                Path(load_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
            
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            
            # Additional memory optimizations
            if device == "cuda":
                model.gradient_checkpointing_enable()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                print(f"✅ Model loaded to GPU: {torch.cuda.get_device_name(0)}")
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"   Memory allocated: {allocated:.2f} GB")
                print(f"   Memory reserved: {reserved:.2f} GB")
            
            return model
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError("Failed to load model after all retries")

# -----------------------------
# UTILITY FUNCTIONS
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
# CUSTOM TRAINER WITH TIMEOUT SUPPORT
# -----------------------------
class SafeTrainer(Trainer):
    """Custom trainer with timeout and hang detection"""
    
    def __init__(self, timeout_monitor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_monitor = timeout_monitor
        self.last_checkpoint_time = time.time()
        
    def training_step(self, model, inputs):
        # Check for timeout before each step
        if self.timeout_monitor and self.timeout_monitor.should_stop():
            print("⏰ Training timeout reached, stopping...")
            raise KeyboardInterrupt()
            
        # Update progress monitor
        if self.timeout_monitor:
            self.timeout_monitor.update_progress()
        
        # Auto-checkpoint every 30 minutes
        current_time = time.time()
        if current_time - self.last_checkpoint_time > CHECKPOINT_INTERVAL:
            print(f"💾 Auto-checkpoint (every {CHECKPOINT_INTERVAL//60} minutes)...")
            try:
                self.save_model(str(OUTPUT_DIR / "checkpoint_latest"))
                self.last_checkpoint_time = current_time
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
        
        return super().training_step(model, inputs)

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 60)
    print("Anthill Forge - Network Optimized Training")
    print("=" * 60)
    
    # Setup timeout handlers
    setup_timeout_handler()
    
    # Initialize timeout monitor
    timeout_monitor = TimeoutMonitor(max_hours=MAX_TRAINING_HOURS)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔍 Using device: {device.upper()}")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find cleaned datasets
    print(f"\n📊 Step 1: Looking for cleaned datasets...")
    
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
        print(f"  • {f.name}: {line_count:,} pairs")
    
    # Step 2: Load and merge datasets
    print("\n📦 Step 2: Loading and merging datasets...")
    dataset = merge_datasets(files)
    
    if len(dataset) == 0:
        print("❌ No data loaded.")
        return 1
    
    print(f"Total training examples: {len(dataset):,}")
    
    # Step 3: Format for training
    print("\n🔄 Step 3: Formatting examples...")
    formatted_dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)
    
    # Step 4: Load tokenizer
    print("\n🔤 Step 4: Loading tokenizer...")
    try:
        # Tokenizer can be loaded from network or cached
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            local_files_only=False  # Allow network access
        )
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("   Trying to load from HuggingFace hub as fallback...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Step 5: Split into train/validation
    print("Splitting dataset (90% train, 10% validation)...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(eval_dataset):,}")
    
    # Step 6: Load model from network
    print("\n🤖 Step 5: Loading model from network...")
    
    try:
        model = load_model_over_network(MODEL_NAME, device=device)
    except Exception as e:
        print(f"❌ Error loading model from network: {e}")
        print("   Falling back to HuggingFace hub...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return 1
    
    # Step 7: Training arguments
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
        fp16=device == "cuda",
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        # Save more frequently for network safety
        save_safetensors=True,
        save_only_model=False,
    )
    
    # Step 8: Create safe trainer
    print("Setting up trainer with timeout monitoring...")
    
    # Start timeout monitor
    timeout_monitor.start()
    
    trainer = SafeTrainer(
        timeout_monitor=timeout_monitor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Step 9: Train with exception handling
    print("\n🚀 Step 7: Starting training...")
    print(f"Training on {len(train_dataset):,} examples")
    print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    estimated_steps = len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS) * NUM_EPOCHS
    print(f"Total steps: ~{estimated_steps:,}")
    print(f"Max training time: {MAX_TRAINING_HOURS} hours")
    
    if device == "cuda":
        print("\n📊 GPU Memory Status:")
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    print("\n" + "-" * 60)
    
    try:
        trainer.train()
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU Out of Memory! Trying to recover...")
        # Try to save current state
        try:
            trainer.save_model(str(OUTPUT_DIR / "oom_recovery"))
            print("Saved recovery checkpoint")
        except:
            pass
        return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Stop timeout monitor
        timeout_monitor.stop()
        print("⏱️  Timeout monitor stopped")
    
    # Step 10: Save model
    print("\n💾 Step 8: Saving model...")
    try:
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        print("✅ Model saved successfully")
    except Exception as e:
        print(f"⚠️  Error saving model: {e}")
        # Try alternative save location
        alt_dir = OUTPUT_DIR / "final_model"
        try:
            trainer.save_model(str(alt_dir))
            tokenizer.save_pretrained(str(alt_dir))
            print(f"✅ Model saved to alternative location: {alt_dir}")
        except:
            print("❌ Failed to save model anywhere!")
    
    # Save metadata
    metadata = {
        "model": MODEL_NAME,
        "device": device,
        "dataset_size": len(dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_time_hours": (time.time() - timeout_monitor.start_time) / 3600 if timeout_monitor.start_time else None,
        "completed": True if not timeout_monitor.should_stop() else False,
    }
    
    try:
        with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except:
        print("⚠️  Could not save metadata")
    
    print(f"\n{'✅' if not timeout_monitor.should_stop() else '⚠️'} Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    if trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        if "loss" in final_log:
            print(f"   Final training loss: {final_log['loss']:.4f}")
        if "eval_loss" in final_log:
            print(f"   Final validation loss: {final_log['eval_loss']:.4f}")
    
    if timeout_monitor.should_stop():
        print(f"\n⚠️  Training stopped due to timeout after {MAX_TRAINING_HOURS} hours")
        print("   Consider increasing MAX_TRAINING_HOURS or reducing dataset size")
    else:
        print(f"\n🎉 Anthill Forge pipeline completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())