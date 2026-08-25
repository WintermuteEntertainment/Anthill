# train_qlora.py — QLoRA fine-tuning for 14B-32B+ models on consumer GPUs
#
# Uses 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (peft) to
# fine-tune large language models on 12-16GB VRAM GPUs with CPU offloading
# for anything that doesn't fit. Designed to be model-agnostic — works with
# any HuggingFace model by using its native chat template.
#
# Usage:
#   python train_qlora.py                                    # defaults from config_qlora.py
#   python train_qlora.py --model Qwen/Qwen2.5-Coder-14B-Instruct
#   python train_qlora.py --model Qwen/Qwen2.5-Coder-14B-Instruct --epochs 1 --max-length 1024
#   python train_qlora.py --skip-merge                       # save adapters only, merge later
#
# After training, run export_gguf.py to convert the merged model to GGUF.

import argparse
import json
import signal
import sys
import time
import threading
from pathlib import Path
from functools import partial

import torch
import psutil
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
try:
    from transformers import HqqConfig
    HQQ_AVAILABLE = True
except ImportError:
    HQQ_AVAILABLE = False
from transformers import TrainerCallback
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

# Import shared configuration (defaults for all settings)
from config_qlora import (
    MODEL_ID, DATA_FOLDER, OUTPUT_DIR, ADAPTER_DIR, MERGED_DIR,
    QLORA_R, QLORA_ALPHA, QLORA_DROPOUT, QLORA_TARGET_MODULES,
    MAX_LENGTH, NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, GRAD_ACCUM_STEPS,
    LEARNING_RATE, WARMUP_RATIO, MAX_GRAD_NORM, MAX_TRAINING_HOURS,
    CHECKPOINT_INTERVAL, HEARTBEAT_INTERVAL,
)


# ─────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────

def parse_args():
    """Parse CLI arguments. Defaults come from config_qlora.py."""
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning for large language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model", default=MODEL_ID,
                   help="HuggingFace model ID or local path")
    # Data
    p.add_argument("--data-folder", type=Path, default=DATA_FOLDER,
                   help="Folder containing *_clean.jsonl files")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH,
                   help="Max token sequence length for training")
    # Training
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    # QLoRA
    p.add_argument("--lora-r", type=int, default=QLORA_R,
                   help="LoRA rank (higher = more capacity, more VRAM)")
    p.add_argument("--lora-alpha", type=int, default=QLORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=QLORA_DROPOUT)
    p.add_argument("--target-modules", type=str, default=None,
                   help="Comma-separated LoRA target module names "
                        "(e.g. q_proj,k_proj,v_proj,o_proj). Overrides "
                        "QLORA_TARGET_MODULES and auto-detect. STRONGLY recommended "
                        "for MoE models (e.g. Qwen3-MoE) so LoRA targets attention "
                        "only, not all 128 experts + the router gate.")
    # Quantization
    p.add_argument("--quant-bits", type=int, default=4, choices=[2, 3, 4],
                   help="Quantization bits: 4=BnB NF4 (default), 3 or 2=HQQ (smaller, fits bigger models)")
    # Output
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--skip-merge", action="store_true",
                   help="Save adapters only, don't merge into base model")
    # Timeout
    p.add_argument("--max-hours", type=float, default=MAX_TRAINING_HOURS)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# TIMEOUT MONITOR — detects hangs and enforces max training time
# (Adapted from train_instruction_model_auto.py)
# ─────────────────────────────────────────────────────────────────

class TimeoutMonitor:
    """Background thread that watches for training hangs and time limits."""

    def __init__(self, max_hours):
        self.max_seconds = max_hours * 3600
        self.start_time = None
        self.last_progress_time = None
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"   Timeout monitor started (max {self.max_seconds / 3600:.1f}h)")

    def _monitor_loop(self):
        while self.is_running and not self._stop_event.is_set():
            now = time.time()
            idle = now - self.last_progress_time

            # Warn if no progress for 30 minutes
            if idle > 1800:
                print(f"\n   WARNING: No progress for {idle / 60:.0f} min!")
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(0) / 1e9
                    print(f"   GPU memory allocated: {alloc:.2f} GB")
                    if alloc < 0.1:
                        print("   GPU appears idle — possible hang!")
                        self._stop_event.set()

            # Enforce max training time
            if now - self.start_time > self.max_seconds:
                print(f"\n   Max training time reached ({self.max_seconds / 3600:.1f}h)")
                self._stop_event.set()
                break

            time.sleep(HEARTBEAT_INTERVAL)

    def tick(self):
        """Call this from the training loop to signal progress."""
        self.last_progress_time = time.time()

    def should_stop(self):
        return self._stop_event.is_set()

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def elapsed_hours(self):
        if self.start_time is None:
            return 0
        return (time.time() - self.start_time) / 3600


# ─────────────────────────────────────────────────────────────────
# SAFE TRAINER — extends Trainer with timeout and auto-checkpoint
# ─────────────────────────────────────────────────────────────────

class SafeTrainer(Trainer):
    """Trainer with automatic timeout checks and periodic checkpointing."""

    def __init__(self, timeout_monitor=None, checkpoint_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_monitor = timeout_monitor
        self.checkpoint_dir = checkpoint_dir
        self._last_checkpoint_time = time.time()

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        # Check if we've exceeded the time limit
        if self.timeout_monitor and self.timeout_monitor.should_stop():
            print("\n   Timeout reached, stopping training gracefully...")
            raise KeyboardInterrupt()

        # Signal that we're making progress (not hung)
        if self.timeout_monitor:
            self.timeout_monitor.tick()

        # Auto-save checkpoint every CHECKPOINT_INTERVAL seconds
        now = time.time()
        if self.checkpoint_dir and (now - self._last_checkpoint_time > CHECKPOINT_INTERVAL):
            print(f"\n   Auto-checkpoint ({CHECKPOINT_INTERVAL // 60}min interval)...")
            try:
                self.save_model(str(self.checkpoint_dir))
                self._last_checkpoint_time = now
            except Exception as e:
                print(f"   Checkpoint save failed: {e}")

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)


# ─────────────────────────────────────────────────────────────────
# STATUS CALLBACK — writes live training metrics to a JSON file
# so the Forge server (forge_server.py) can serve them to the
# Chrome extension dashboard in real-time.
# ─────────────────────────────────────────────────────────────────

class ForgeStatusCallback(TrainerCallback):
    """Writes a training_status.json file every N logging steps."""

    def __init__(self, output_dir, total_epochs):
        self.status_path = Path(output_dir) / "training_status.json"
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called every logging_steps — write current metrics to disk."""
        elapsed = time.time() - self.start_time
        status = {
            "phase": f"Training epoch {int(state.epoch)}/{self.total_epochs}",
            "message": f"Step {state.global_step} / {state.max_steps}",
            "current_step": state.global_step,
            "total_steps": state.max_steps,
            "current_epoch": int(state.epoch),
            "total_epochs": self.total_epochs,
            "loss": logs.get("loss") if logs else None,
            "learning_rate": logs.get("learning_rate") if logs else None,
            "elapsed_seconds": round(elapsed),
        }

        # Calculate ETA based on average step speed
        if state.global_step > 0:
            secs_per_step = elapsed / state.global_step
            remaining_steps = state.max_steps - state.global_step
            status["eta"] = round(secs_per_step * remaining_steps)
            status["samples_per_second"] = logs.get("train_samples_per_second") if logs else None

        # Add GPU memory info if available
        if torch.cuda.is_available():
            status["vram_used"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            status["vram_total"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

        try:
            with open(self.status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass  # non-critical — don't interrupt training

    def on_train_end(self, args, state, control, **kwargs):
        """Mark training as finished in the status file."""
        elapsed = time.time() - self.start_time
        final_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    final_loss = entry["loss"]
                    break

        status = {
            "phase": "Complete",
            "message": "Training finished",
            "current_step": state.global_step,
            "total_steps": state.max_steps,
            "current_epoch": self.total_epochs,
            "total_epochs": self.total_epochs,
            "loss": final_loss,
            "elapsed_seconds": round(elapsed),
        }

        try:
            with open(self.status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
# QUANTIZATION CONFIG — 4-bit NF4 or 3/2-bit HQQ
# ─────────────────────────────────────────────────────────────────

def get_quant_config(quant_bits=4):
    """
    Build quantization config.
    - 4-bit: BitsAndBytes NF4 (best quality, standard QLoRA)
    - 3-bit or 2-bit: HQQ (smaller footprint, fits bigger models on less VRAM)
    """
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"   Compute dtype: {compute_dtype}")
    print(f"   Quantization: {quant_bits}-bit {'BnB NF4' if quant_bits == 4 else 'HQQ'}")

    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality 4-bit
            bnb_4bit_compute_dtype=compute_dtype, # math happens in bf16/fp16
            bnb_4bit_use_double_quant=True,       # quantize the quantization constants too
        )
    else:
        # HQQ for 3-bit or 2-bit — smaller model footprint
        if not HQQ_AVAILABLE:
            print("   ERROR: HQQ not available. Install with: pip install hqq")
            print("   Falling back to 4-bit BnB.")
            return get_quant_config(4)
        return HqqConfig(
            nbits=quant_bits,
            group_size=64,
            quant_zero=False,
            quant_scale=False,
            axis=0,                # row-wise quantization (faster, avoids stall)
        )


# ─────────────────────────────────────────────────────────────────
# LORA CONFIG — which layers to train and how
# ─────────────────────────────────────────────────────────────────

def find_all_linear_names(model):
    """
    Walk the model and return the names of all linear layers
    (excluding lm_head). This makes the script model-agnostic —
    it doesn't need to know the architecture-specific layer names
    (q_proj, k_proj, etc.). Works for Qwen, Llama, Phi, Mistral, etc.

    Detects nn.Linear, BnB 4-bit layers, and HQQ quantized layers.
    """
    # Import quantized layer types we might encounter
    linear_types = [torch.nn.Linear]
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        linear_types.extend([Linear4bit, Linear8bitLt])
    except ImportError:
        pass
    try:
        from hqq.core.quantize import HQQLinear
        linear_types.append(HQQLinear)
    except ImportError:
        pass
    linear_types = tuple(linear_types)

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            # Get the short name (last component). Skip the output head AND the
            # MoE router gate (named "gate"): the router must stay full-precision
            # and untrained, or expert routing destabilizes during fine-tuning.
            short = name.split(".")[-1]
            if short not in ("lm_head", "gate"):
                names.add(short)

    # Fallback: if nothing found, use standard Qwen/Llama layer names
    if not names:
        print("   WARNING: Auto-detection found no linear layers, using standard names")
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    # Deduplicate and sort for reproducibility
    return sorted(names)


def get_lora_config(model, args):
    """
    Build LoraConfig targeting all linear layers (or specific ones
    if --target-modules is passed, or QLORA_TARGET_MODULES is set in config).

    Precedence: CLI --target-modules > config QLORA_TARGET_MODULES > auto-detect.
    """
    if getattr(args, "target_modules", None):
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    elif QLORA_TARGET_MODULES:
        target_modules = QLORA_TARGET_MODULES
    else:
        target_modules = find_all_linear_names(model)

    print(f"   LoRA target modules: {target_modules}")
    print(f"   LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",                    # don't train biases (standard for QLoRA)
        task_type=TaskType.CAUSAL_LM,
    )


# ─────────────────────────────────────────────────────────────────
# DATASET FORMATTING — uses model's native chat template
# ─────────────────────────────────────────────────────────────────

def format_with_chat_template(example, tokenizer):
    """
    Format a prompt/completion pair using the model's own chat template.

    This is the KEY difference from the old Forge scripts, which used:
        f"Human: {prompt}\\n\\nAssistant: {completion}"

    That format only works for models trained on it (like Anthropic's Claude).
    Qwen2.5 expects ChatML, Llama expects [INST] tags, etc.

    By calling tokenizer.apply_chat_template(), we automatically get the
    right format for whatever model we're fine-tuning. This makes the
    script truly model-agnostic.
    """
    messages = [
        {"role": "user", "content": example["prompt"].strip()},
        {"role": "assistant", "content": example["completion"].strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # don't append a trailing prompt marker
    )
    return {"text": text}


# ─────────────────────────────────────────────────────────────────
# DATA LOADING — same JSONL format as existing Loom pipeline
# ─────────────────────────────────────────────────────────────────

def find_dataset_files(data_folder):
    """Find all cleaned JSONL files produced by Anthill Loom."""
    files = sorted(data_folder.rglob("*clean*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No *clean*.jsonl files found in {data_folder}\n"
            f"Run anthill-loom/run_chatgpt_pipeline.bat first."
        )
    return files


def load_and_merge_datasets(files):
    """Load multiple JSONL files and concatenate into one dataset."""
    datasets_list = []
    for f in files:
        print(f"   Loading {f.name}...")
        try:
            ds = load_dataset("json", data_files=str(f), split="train")
            # Filter out empty pairs
            ds = ds.filter(lambda x: x.get("prompt") and x.get("completion"))
            datasets_list.append(ds)
            print(f"      {len(ds)} pairs")
        except Exception as e:
            print(f"      Error: {e}")
    if not datasets_list:
        raise RuntimeError("No valid datasets loaded")
    return concatenate_datasets(datasets_list)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING — 4-bit quantized with LoRA adapters
# ─────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id, quant_config, lora_args, quant_bits=4):
    """
    Load a model with quantization, prepare it for training,
    and attach LoRA adapters. Returns (model, tokenizer).

    All bit widths use device_map="cuda:0" (no CPU offload).
    CPU offload is incompatible with 4/8-bit training in accelerate.
    """
    print(f"\n   Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    # Many models don't set a pad token — use eos as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Vocab size: {tokenizer.vocab_size}")

    print(f"   Loading model in {quant_bits}-bit (this may take several minutes)...")
    start = time.time()

    # Force everything onto GPU — no CPU offload.
    # CPU offload causes "can't train 4/8-bit with CPU offload" errors.
    # At 4-bit NF4 + double quant, 32B ≈ 14-15GB (tight on 16GB, but viable).
    # At 3/2-bit HQQ, 32B ≈ 10-12GB (comfortable on 16GB).
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"   GPU VRAM total: {gpu_mem/1e9:.1f}GB")
        print(f"   Strategy: device_map='cuda:0' (no CPU offload)")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="cuda:0",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    elapsed = time.time() - start
    print(f"   Model loaded in {elapsed:.0f}s")

    # Report memory usage
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU VRAM: {alloc:.1f}GB used / {total:.1f}GB total")
    ram = psutil.virtual_memory()
    print(f"   System RAM: {ram.used / 1e9:.1f}GB used / {ram.total / 1e9:.1f}GB total")

    # Prepare for k-bit training (freezes base weights, enables gradient for LoRA)
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing (trades compute for memory — essential)
    model.gradient_checkpointing_enable()

    # Attach LoRA adapters
    lora_config = get_lora_config(model, lora_args)
    model = get_peft_model(model, lora_config)

    # Report trainable parameters
    trainable, total_params = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total_params
    print(f"   Trainable params: {trainable:,} / {total_params:,} ({pct:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# MERGE — bake LoRA weights back into the base model
# ─────────────────────────────────────────────────────────────────

def merge_and_save(adapter_dir, model_id, merged_dir):
    """
    Reload the base model in fp16 (on CPU), load the trained LoRA adapter,
    merge them together, and save the result as a full HuggingFace model.

    This requires enough system RAM to hold the model in fp16:
      - 14B → ~28GB RAM
      - 32B → ~64GB RAM
    With 64-128GB DDR5 available, this is fine.

    The merged model can then be converted to GGUF via export_gguf.py.
    """
    print(f"\n   Merging LoRA adapters into base model...")
    print(f"   This loads the full model in fp16 on CPU — needs lots of RAM.")

    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Reload base model in fp16 on CPU (no quantization this time)
    print(f"   Loading base model {model_id} in fp16...")
    start = time.time()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",             # keep everything on CPU for merge
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"   Base model loaded in {time.time() - start:.0f}s")

    ram = psutil.virtual_memory()
    print(f"   RAM after base load: {ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB")

    # Load the trained adapter on top
    print(f"   Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # Merge adapter weights into the base model and drop the adapter wrapper
    print(f"   Merging weights...")
    model = model.merge_and_unload()

    # Save the merged model
    print(f"   Saving merged model to {merged_dir}...")
    model.save_pretrained(str(merged_dir))

    # Save tokenizer alongside the model (needed for GGUF conversion)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"   Merged model saved to {merged_dir}")
    return merged_dir


# ─────────────────────────────────────────────────────────────────
# SIGNAL HANDLERS — graceful shutdown on Ctrl+C
# ─────────────────────────────────────────────────────────────────

def setup_signal_handlers():
    """Install signal handlers for clean interruption."""
    def handler(sig, frame):
        print(f"\n   Received signal {sig}, shutting down gracefully...")
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("  ANTHILL FORGE — QLoRA Fine-Tuning Pipeline")
    print("=" * 65)
    print(f"  Model:          {args.model}")
    print(f"  Data:           {args.data_folder}")
    print(f"  Output:         {args.output_dir}")
    print(f"  LoRA rank:      {args.lora_r}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Max length:     {args.max_length}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Skip merge:     {args.skip_merge}")
    print("=" * 65)

    setup_signal_handlers()

    # ── Step 1: Check hardware ──────────────────────────────────
    print("\n Step 1: Hardware check")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("   WARNING: No GPU found. QLoRA training on CPU is impractical.")
        print("   Exiting. Use a machine with an NVIDIA GPU.")
        return 1

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {name} ({vram:.1f} GB VRAM)")

    if torch.cuda.is_bf16_supported():
        print("   bfloat16: supported")
    else:
        print("   bfloat16: NOT supported, using float16")

    ram = psutil.virtual_memory()
    print(f"   System RAM: {ram.total / 1e9:.0f} GB")

    # ── Step 2: Create output directories ───────────────────────
    adapter_dir = args.output_dir / "adapters"
    merged_dir = args.output_dir / "merged"
    checkpoint_dir = args.output_dir / "checkpoint_latest"
    for d in [args.output_dir, adapter_dir, merged_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 3: Load datasets ───────────────────────────────────
    print("\n Step 2: Loading datasets")

    files = find_dataset_files(args.data_folder)
    print(f"   Found {len(files)} dataset file(s):")
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            count = sum(1 for _ in fh)
        print(f"      {f.name}: {count:,} pairs")

    dataset = load_and_merge_datasets(files)
    print(f"   Total training pairs: {len(dataset):,}")

    # ── Step 4: Load model + tokenizer ───────────────────────────
    print(f"\n Step 3: Loading model in {args.quant_bits}-bit quantization")

    quant_config = get_quant_config(args.quant_bits)
    model, tokenizer = load_model_and_tokenizer(args.model, quant_config, args, args.quant_bits)

    # ── Step 5: Format dataset with native chat template ────────
    print("\n Step 4: Formatting with model's chat template")

    # Show a sample so the user can verify the format is correct
    sample = format_with_chat_template(dataset[0], tokenizer)
    print(f"   Sample formatted text (first 300 chars):")
    print(f"   ---")
    print(f"   {sample['text'][:300]}")
    print(f"   ---")

    # Apply to full dataset
    format_fn = partial(format_with_chat_template, tokenizer=tokenizer)
    formatted = dataset.map(format_fn, remove_columns=dataset.column_names)

    # ── Step 6: Tokenize ────────────────────────────────────────
    print("\n Step 5: Tokenizing")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # dynamic padding saves memory vs padding="max_length"
        )

    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])

    # Split 90/10 train/eval
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    print(f"   Train: {len(split['train']):,}")
    print(f"   Eval:  {len(split['test']):,}")

    # ── Step 7: Configure training ──────────────────────────────
    print("\n Step 6: Configuring training")

    use_bf16 = torch.cuda.is_bf16_supported()
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(split["train"]) // effective_batch)
    total_steps = steps_per_epoch * args.epochs

    print(f"   Optimizer: paged_adamw_8bit (memory-efficient)")
    print(f"   LR schedule: cosine, warmup {WARMUP_RATIO * 100:.0f}%")
    print(f"   Precision: {'bfloat16' if use_bf16 else 'float16'}")
    print(f"   Steps/epoch: {steps_per_epoch:,}")
    print(f"   Total steps: ~{total_steps:,}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Use paged 8-bit AdamW — prevents OOM during gradient spikes
        # by paging optimizer states to CPU when GPU is full
        optim="paged_adamw_8bit",

        # Precision — bf16 preferred (better range than fp16, no grad scaling issues)
        bf16=use_bf16,
        fp16=not use_bf16,

        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=False,  # we save adapters manually

        # Logging
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=10,
        report_to="none",              # no wandb/hf hub

        # Memory optimization
        gradient_checkpointing=True,
        dataloader_pin_memory=False,   # safer with CPU offloading
        dataloader_num_workers=0,      # avoid Windows multiprocessing issues
        remove_unused_columns=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
    )

    # ── Step 8: Train ───────────────────────────────────────────
    print(f"\n Step 7: Training")
    print(f"   Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Max time: {args.max_hours}h")
    print("-" * 65)

    timeout = TimeoutMonitor(args.max_hours)
    timeout.start()

    # Create the status callback so the Forge extension can show live metrics
    status_callback = ForgeStatusCallback(
        output_dir=args.output_dir,
        total_epochs=args.epochs,
    )

    trainer = SafeTrainer(
        timeout_monitor=timeout,
        checkpoint_dir=checkpoint_dir,
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # causal LM, not masked LM
        ),
        callbacks=[status_callback],
    )

    interrupted = False
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n   GPU OUT OF MEMORY!")
        print("   Try: --batch-size 1, --max-length 1024, or a smaller model")
        # Save what we have before exiting
        try:
            model.save_pretrained(str(adapter_dir / "oom_recovery"))
        except Exception:
            pass
        return 1
    except KeyboardInterrupt:
        print("\n   Training interrupted — saving current state...")
        interrupted = True
    except Exception as e:
        print(f"\n   Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        timeout.stop()

    print("-" * 65)
    print(f"   {'Interrupted' if interrupted else 'Completed'} at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Elapsed: {timeout.elapsed_hours:.1f}h")

    # Report final metrics
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        if "loss" in last:
            print(f"   Final training loss: {last['loss']:.4f}")
        if "eval_loss" in last:
            print(f"   Final eval loss: {last['eval_loss']:.4f}")

    # ── Step 9: Save adapters ───────────────────────────────────
    print(f"\n Step 8: Saving LoRA adapters to {adapter_dir}")

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("   Adapters saved.")

    # ── Step 10: Merge (optional) ───────────────────────────────
    if not args.skip_merge:
        print(f"\n Step 9: Merging adapters into base model")
        try:
            merge_and_save(adapter_dir, args.model, merged_dir)
        except Exception as e:
            print(f"   Merge failed: {e}")
            print("   Adapters are still saved — you can merge later with:")
            print(f"   python train_qlora.py --skip-merge  (already done)")
            print(f"   Or manually load adapter from {adapter_dir}")
    else:
        print(f"\n   Skipping merge (--skip-merge). Adapters at {adapter_dir}")
        print(f"   To merge later, load the adapter and call merge_and_unload().")

    # ── Step 11: Save metadata ──────────────────────────────────
    metadata = {
        "model": args.model,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_hours": round(timeout.elapsed_hours, 2),
        "interrupted": interrupted,
        "dataset_size": len(dataset),
        "train_size": len(split["train"]),
        "eval_size": len(split["test"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": QLORA_TARGET_MODULES or "auto (all linear)",
        "quantization": "4-bit NF4",
        "precision": "bfloat16" if use_bf16 else "float16",
        "merged": not args.skip_merge,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if not args.skip_merge else None,
    }

    meta_path = args.output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n   Metadata saved to {meta_path}")

    # ── Done ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if not args.skip_merge:
        print(f"  DONE! Merged model at: {merged_dir}")
        print(f"  Next: python export_gguf.py")
    else:
        print(f"  DONE! Adapters at: {adapter_dir}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
