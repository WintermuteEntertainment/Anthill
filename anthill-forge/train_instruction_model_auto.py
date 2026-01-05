#train_instruction_model_auto_fp16_int8.py

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
import json

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/phi-2"   # or your chosen instruction-tuned base
DATA_FOLDER = Path("datasets/processed")
OUTPUT_DIR = Path("model_out")
MAX_LENGTH = 2048

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5

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
        ds = load_dataset("json", data_files=str(f))["train"]
        datasets_list.append(ds)
    return concatenate_datasets(datasets_list)

def format_for_training(example):
    """Turn prompt/completion pair into text for instruction tuning"""
    return {
        "text": f"User:\n{example['prompt']}\n\nAssistant:\n{example['completion']}"
    }

# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Scanning for cleaned datasets in {DATA_FOLDER}")
    files = get_all_jsonl_files(DATA_FOLDER)
    if not files:
        raise RuntimeError("No cleaned JSONL files found. Run preprocessing first.")

    print(f"Found {len(files)} files. Loading and concatenating...")
    dataset = merge_datasets(files)
    print(f"Total examples: {len(dataset)}")

    print("Formatting examples for instruction tuning...")
    dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")

if __name__ == "__main__":
    main()
