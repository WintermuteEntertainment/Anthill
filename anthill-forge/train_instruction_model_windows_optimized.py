# train_instruction_model_windows_optimized.py
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

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = Path("//BigBlackBox/a/gertrude_phi2_finetune/final_model")
DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = Path("C:/anthill_forge_output_optimized")

MAX_LENGTH = 1024

# Hyperparameters
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-5

# Data loading
NUM_WORKERS = 4  # Use 4 workers for data loading

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 70)
    print("ANTHILL FORGE - WINDOWS OPTIMIZED")
    print("=" * 70)
    
    # Set environment variable for tokenizers
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\nLoading model...")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.gradient_checkpointing_enable()
    
    # Load datasets
    print("\nLoading datasets...")
    files = list(DATA_FOLDER.rglob("*_clean.jsonl"))
    datasets_list = []
    for f in files:
        ds = load_dataset("json", data_files=str(f))["train"]
        datasets_list.append(ds)
    dataset = concatenate_datasets(datasets_list)
    
    # Format and tokenize
    def format_example(ex):
        return {"text": f"Human: {ex['prompt']}\n\nAssistant: {ex['completion']}"}
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    tokenized_dataset = formatted_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Split
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Training arguments
    effective_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print(f"\nEffective batch size: {effective_batch}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        
        # Save and evaluate during training
        save_strategy="steps",
        save_steps=100,  # Save every 100 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        
        # Logging
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        
        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,
        fp16_full_eval=False,
        
        # Performance
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        
        # Data loading
        dataloader_pin_memory=True,  # Faster data transfer to GPU
        dataloader_num_workers=NUM_WORKERS,  # Multi-worker data loading
        
        # Other
        remove_unused_columns=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    print(f"\nTraining completed in {time.time() - start_time:.0f} seconds")
    
    # Save final model
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
