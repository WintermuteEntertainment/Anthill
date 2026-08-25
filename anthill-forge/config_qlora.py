# config_qlora.py — Shared configuration for QLoRA training and GGUF export
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Model ────────────────────────────────────────────────────────
# HuggingFace model ID or local path. Override via CLI: --model <id>
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ── Paths ────────────────────────────────────────────────────────
DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR = Path("C:/anthill_forge_output/qlora")
ADAPTER_DIR = OUTPUT_DIR / "adapters"
MERGED_DIR = OUTPUT_DIR / "merged"

# Network paths (BigBlackBox)
GGUF_OUTPUT_DIR = Path("//BigBlackBox/a/LLAMA/models")
LLAMA_CPP_CONVERT = Path("//BigBlackBox/a/llama.cpp/convert_hf_to_gguf.py")
LLAMA_CPP_QUANTIZE = Path("//BigBlackBox/a/llama.cpp/build/bin/llama-quantize")

# ── QLoRA Settings ───────────────────────────────────────────────
QLORA_R = 64                 # LoRA rank (higher = more capacity, more VRAM)
QLORA_ALPHA = 16             # scaling factor (alpha / r = scaling)
QLORA_DROPOUT = 0.05
QLORA_TARGET_MODULES = None  # None = auto-detect all linear layers

# ── Training Hyperparameters ─────────────────────────────────────
MAX_LENGTH = 2048
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16        # effective batch = BATCH * ACCUM = 16
LEARNING_RATE = 2e-4          # 10x higher than full fine-tune (standard for LoRA)
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 0.3
MAX_TRAINING_HOURS = 24
CHECKPOINT_INTERVAL = 1800    # auto-save every 30 min
HEARTBEAT_INTERVAL = 300      # hang detection every 5 min

# ── GGUF Export ──────────────────────────────────────────────────
QUANT_TYPE = "Q8_0"           # Q4_K_M, Q5_K_M, Q8_0, F16
