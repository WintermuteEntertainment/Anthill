# resume_training.py
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

# -----------------------------
# CONFIG (MATCH MAIN SCRIPT)
# -----------------------------
MODEL_PATH = Path("C:/anthill_forge_output")  # checkpoint parent or interrupted save
CHECKPOINT_PATH = MODEL_PATH / "checkpoint-XXXX"  # <-- CHANGE THIS
DATA_FOLDER = Path("datasets/processed")
OUTPUT_DIR = Path("C:/anthill_forge_output_resumed")

MAX_LENGTH = 1024
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-5

# -----------------------------
# DATA
# -----------------------------
def get_all_jsonl_files(folder):
    return list(folder.rglob("*_clean.jsonl"))

def merge_datasets(files):
    datasets_list = []
    for f in files:
        ds = load_dataset("json", data_files=str(f))["train"]
        datasets_list.append(ds)
    return concatenate_datasets(datasets_list)

def format_for_training(example):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    return {"text": f"Human: {prompt}\n\nAssistant: {completion}"}

# -----------------------------
# MAIN
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    files = get_all_jsonl_files(DATA_FOLDER)
    dataset = merge_datasets(files)
    dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    model.gradient_checkpointing_enable()

    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,

        save_strategy="steps",
        save_steps=500,                 # IMPORTANT for resuming again
        save_total_limit=2,

        logging_steps=10,
        report_to="none",

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    # Resume training
    trainer.train(resume_from_checkpoint=str(CHECKPOINT_PATH))

    # Save final
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

if __name__ == "__main__":
    main()
