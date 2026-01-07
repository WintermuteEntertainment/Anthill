# train_minimal_network.py
import torch
from pathlib import Path
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

print("=" * 60)
print("MINIMAL NETWORK TRAINING - Reducing network traffic")
print("=" * 60)

# Use local cache if possible, fallback to network
LOCAL_CACHE = Path("C:/model_cache/gertrude_phi2/final_model")
NETWORK_PATH = Path("Z:/gertrude_phi2_finetune/final_model")

if LOCAL_CACHE.exists():
    MODEL_PATH = LOCAL_CACHE
    print(f"✅ Using LOCAL cache: {MODEL_PATH}")
else:
    MODEL_PATH = NETWORK_PATH
    print(f"⚠️ Using NETWORK path (will be slow): {MODEL_PATH}")

# Load ONLY what's absolutely necessary
print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    trust_remote_code=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

print("\n🤖 Loading model (this may take several minutes over network)...")
start = time.time()

# Load with minimal network options
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    use_cache=False,  # Required for gradient checkpointing
)

print(f"✅ Model loaded in {time.time()-start:.1f} seconds")

# Create a tiny test dataset (so we can see if training works)
test_data = [
    {"prompt": "What is 2+2?", "completion": "2+2 equals 4."},
    {"prompt": "Capital of France?", "completion": "The capital of France is Paris."},
]

# Format and tokenize
def format_example(ex):
    return {"text": f"Human: {ex['prompt']}\n\nAssistant: {ex['completion']}"}

formatted = [format_example(d) for d in test_data]
tokenized = tokenizer(
    [d["text"] for d in formatted],
    truncation=True,
    padding="max_length",
    max_length=512
)

# Convert to dataset format
import torch
class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = TinyDataset(tokenized)

# Minimal training args - just 1 epoch, small steps
training_args = TrainingArguments(
    output_dir="./test_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,  # Important for network
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\n🚀 Starting MINIMAL training (2 examples, 1 epoch)...")
print("   This will test if training works without hanging")
print("   Monitor GPU usage and network activity")

try:
    trainer.train()
    print("\n✅ SUCCESS! Minimal training completed without hanging")
    print("   The issue was likely the dataset size or processing")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
