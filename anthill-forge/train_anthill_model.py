# train_anthill_model.py
import torch
from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import sys
import subprocess
import json

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = r"\\Bigblackbox\a\gertrude_phi2_finetune\final_model"   # Your fine-tuned phi-2
DATA_FOLDER = Path(r"x:\anthill\anthill\anthill-loom\datasets\processed")
OUTPUT_DIR = Path("anthill_forge_output")
MAX_LENGTH = 2048

# Training hyperparameters
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2  # Adjust based on your GPU memory
GRAD_ACCUM_STEPS = 4        # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def find_and_preprocess_data():
    """Find cleaned JSONL files, or create them if they don't exist"""
    # Look for any JSON files in the datasets/raw folder
    raw_folder = DATA_FOLDER.parent / "raw"
    processed_folder = DATA_FOLDER
    
    # Create folders if they don't exist
    processed_folder.mkdir(parents=True, exist_ok=True)
    
    # Check for cleaned files
    clean_files = list(processed_folder.rglob("*_clean.jsonl"))
    
    if clean_files:
        print(f"Found {len(clean_files)} cleaned dataset(s):")
        for f in clean_files:
            with open(f, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            print(f"  • {f.name}: {line_count} pairs")
        return clean_files
    
    # If no cleaned files, check for raw JSON files
    raw_files = list(raw_folder.rglob("chatgpt_conversations_*.json"))
    
    if not raw_files:
        raise RuntimeError(f"No raw JSON files found in {raw_folder}")
    
    print(f"Found {len(raw_files)} raw JSON file(s), preprocessing...")
    
    clean_files = []
    for raw_file in raw_files:
        # Create output filename
        clean_name = f"{raw_file.stem}_clean.jsonl"
        clean_path = processed_folder / clean_name
        
        # Run preprocessing pipeline
        print(f"Processing {raw_file.name}...")
        
        # Step 1: Extract pairs
        pairs_path = processed_folder / f"{raw_file.stem}_pairs.jsonl"
        
        # Import your extraction function
        from prepare_datasets_parallel import main as extract_pairs
        extract_pairs(str(raw_file), str(pairs_path))
        
        # Step 2: Clean and deduplicate
        from dedupe_and_filter import main as clean_pairs
        clean_pairs(str(pairs_path), str(clean_path))
        
        clean_files.append(clean_path)
        
        # Optional: Clean up intermediate files
        pairs_path.unlink(missing_ok=True)
    
    return clean_files

def load_and_merge_datasets(files):
    """Load multiple JSONL files and concatenate them"""
    datasets_list = []
    
    for file_path in files:
        print(f"Loading {file_path.name}...")
        try:
            # Load dataset from JSONL
            dataset = load_dataset('json', data_files=str(file_path), split='train')
            
            # Count examples
            print(f"  • Contains {len(dataset)} examples")
            
            # Filter out any empty prompts or completions (safety check)
            dataset = dataset.filter(lambda x: x['prompt'] and x['completion'])
            
            datasets_list.append(dataset)
            
        except Exception as e:
            print(f"  • Error loading {file_path}: {e}")
            continue
    
    if not datasets_list:
        raise RuntimeError("No valid datasets could be loaded")
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"Total training examples: {len(combined_dataset)}")
    
    return combined_dataset

def format_for_training(example):
    """Format prompt/completion pairs for instruction tuning"""
    # Create a clean instruction format
    prompt = example['prompt'].strip()
    completion = example['completion'].strip()
    
    # Use a simple instruction format
    formatted_text = f"Human: {prompt}\n\nAssistant: {completion}"
    
    return {"text": formatted_text}

# -----------------------------
# MAIN TRAINING FUNCTION
# -----------------------------
def main():
    print("=" * 60)
    print("Anthill Forge - Instruction Tuning Pipeline")
    print("=" * 60)
    
    # Step 1: Find or create cleaned datasets
    print("\n📊 Step 1: Preparing datasets...")
    dataset_files = find_and_preprocess_data()
    
    if not dataset_files:
        raise RuntimeError("No datasets available for training")
    
    # Step 2: Load and merge datasets
    print("\n📦 Step 2: Loading and formatting data...")
    dataset = load_and_merge_datasets(dataset_files)
    
    # Step 3: Format for training
    print("Formatting examples for instruction tuning...")
    formatted_dataset = dataset.map(
        format_for_training,
        remove_columns=dataset.column_names
    )
    
    # Step 4: Load tokenizer
    print("\n🔤 Step 3: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 5: Tokenize dataset
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Step 6: Split into train/validation (90/10 split)
    print("Splitting into train/validation sets...")
    split_dataset = tokenized_dataset.train_test_split(
        test_size=0.1,
        seed=42
    )
    
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Validation examples: {len(split_dataset['test'])}")
    
    # Step 7: Load model
    print("\n🤖 Step 4: Loading model...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Move to device if not using auto device map
    if device == "cuda" and model.device.type != "cuda":
        model = model.to(device)
    
    # Step 8: Set up training arguments
    print("\n⚙️ Step 5: Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=OUTPUT_DIR / "logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",  # Use mixed precision on CUDA
        report_to="none",  # Disable wandb/huggingface hub reporting
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Save memory
        optim="adamw_torch",
    )
    
    # Step 9: Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Step 10: Train!
    print("\n🚀 Step 6: Starting training...")
    print(f"Training on {len(split_dataset['train'])} examples")
    print(f"Validating on {len(split_dataset['test'])} examples")
    print(f"Total steps: ~{len(split_dataset['train']) // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS) * NUM_EPOCHS}")
    
    trainer.train()
    
    # Step 11: Save the model
    print("\n💾 Step 7: Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metadata
    metadata = {
        "model": MODEL_NAME,
        "dataset_size": len(dataset),
        "train_size": len(split_dataset["train"]),
        "val_size": len(split_dataset["test"]),
        "epochs": NUM_EPOCHS,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    }
    
    with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Final loss: {trainer.state.log_history[-1]['loss']:.4f}")
    print(f"   Training time: {trainer.state.log_history[-1]['train_runtime']:.0f} seconds")
    
    # Optional: Run final evaluation
    print("\n📈 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Validation loss: {eval_results['eval_loss']:.4f}")

if __name__ == "__main__":
    main()
