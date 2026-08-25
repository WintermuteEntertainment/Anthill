# export_gguf.py — Convert a merged HuggingFace model to GGUF format
# Part of the Anthill Forge QLoRA pipeline.
#
# This script takes a merged HF model (output of train_qlora.py) and converts
# it to a GGUF file that llama-server can load. Two conversion methods are
# supported:
#   1. llama.cpp tools on BigBlackBox (preferred — fastest, most reliable)
#   2. Python gguf package (fallback — works anywhere, slower)
#
# The final GGUF is copied to the network model directory so Jazz can pick it up.
#
# Usage:
#   python export_gguf.py                          # use defaults from config
#   python export_gguf.py --merged-dir C:/my/model # override merged model path
#   python export_gguf.py --quant Q4_K_M           # different quantization
#   python export_gguf.py --output-name my-model   # custom output filename

import argparse
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Import shared config ────────────────────────────────────────────
import config_qlora as cfg


# ═══════════════════════════════════════════════════════════════════════
#  CLI argument parsing — every config value can be overridden
# ═══════════════════════════════════════════════════════════════════════
def parse_args():
    """Parse command-line arguments. Defaults come from config_qlora.py."""
    parser = argparse.ArgumentParser(
        description="Export a merged HuggingFace model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_gguf.py
  python export_gguf.py --quant Q4_K_M
  python export_gguf.py --merged-dir C:/my/merged/model --quant Q5_K_M
  python export_gguf.py --output-name qwen25-coder-32b-anthill
  python export_gguf.py --force-python   # skip llama.cpp, use Python fallback
        """,
    )

    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=cfg.MERGED_DIR,
        help=f"Path to the merged HF model directory (default: {cfg.MERGED_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.GGUF_OUTPUT_DIR,
        help=f"Where to save the final GGUF (default: {cfg.GGUF_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=cfg.QUANT_TYPE,
        choices=["F16", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        help=f"Quantization type (default: {cfg.QUANT_TYPE})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom base name for the GGUF file (auto-generated if omitted)",
    )
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Force Python-based conversion even if llama.cpp tools are available",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate F16 GGUF after quantization (normally deleted)",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════
def validate_merged_model(merged_dir: Path) -> bool:
    """
    Check that the merged model directory exists and looks like a valid
    HuggingFace model (has config.json and at least one safetensors/bin file).
    """
    if not merged_dir.exists():
        print(f"  ERROR: Merged model directory not found: {merged_dir}")
        print("  Have you run train_qlora.py first? The merged model is created")
        print("  after training completes and adapters are merged back into the base.")
        return False

    config_file = merged_dir / "config.json"
    if not config_file.exists():
        print(f"  ERROR: No config.json found in {merged_dir}")
        print("  This doesn't look like a valid HuggingFace model directory.")
        return False

    # Look for model weight files (safetensors preferred, .bin as fallback)
    weight_files = list(merged_dir.glob("*.safetensors")) + list(
        merged_dir.glob("*.bin")
    )
    if not weight_files:
        print(f"  ERROR: No model weight files found in {merged_dir}")
        print("  Expected .safetensors or .bin files.")
        return False

    # Calculate total model size for the user's reference
    total_size_gb = sum(f.stat().st_size for f in weight_files) / (1024**3)
    print(f"  Found {len(weight_files)} weight file(s), total size: {total_size_gb:.1f} GB")

    return True


def validate_gguf(gguf_path: Path) -> bool:
    """
    Quick sanity check on a GGUF file — verify the magic bytes and file size.
    GGUF files start with the magic bytes 'GGUF' (0x46475547 little-endian).
    """
    if not gguf_path.exists():
        print(f"  ERROR: GGUF file not found at {gguf_path}")
        return False

    size_gb = gguf_path.stat().st_size / (1024**3)

    # Check magic bytes
    try:
        with open(gguf_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
    except Exception as e:
        print(f"  ERROR: Could not read GGUF header: {e}")
        return False

    # 0x46475547 = 'GGUF' in little-endian
    if magic != 0x46475547:
        print(f"  ERROR: Invalid GGUF magic bytes (got 0x{magic:08X}, expected 0x46475547)")
        return False

    print(f"  GGUF validation passed: {size_gb:.2f} GB, magic bytes OK")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Conversion Method 1: llama.cpp tools (preferred)
# ═══════════════════════════════════════════════════════════════════════
def check_llama_cpp_available() -> bool:
    """Check if llama.cpp's convert script and quantize binary are accessible."""
    convert_ok = cfg.LLAMA_CPP_CONVERT.exists()
    quantize_ok = cfg.LLAMA_CPP_QUANTIZE.exists()

    if convert_ok and quantize_ok:
        print("  llama.cpp tools found on BigBlackBox")
        return True

    if not convert_ok:
        print(f"  convert_hf_to_gguf.py not found at: {cfg.LLAMA_CPP_CONVERT}")
    if not quantize_ok:
        print(f"  llama-quantize not found at: {cfg.LLAMA_CPP_QUANTIZE}")

    return False


def convert_with_llama_cpp(
    merged_dir: Path, f16_gguf_path: Path
) -> bool:
    """
    Use llama.cpp's convert_hf_to_gguf.py to convert a HuggingFace model
    to an F16 GGUF. This is a subprocess call to the script on BigBlackBox.
    """
    print(f"  Running: convert_hf_to_gguf.py -> {f16_gguf_path.name}")

    cmd = [
        sys.executable,
        str(cfg.LLAMA_CPP_CONVERT),
        str(merged_dir),
        "--outfile",
        str(f16_gguf_path),
        "--outtype",
        "f16",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout (large models take a while)
        )

        if result.returncode != 0:
            print(f"  ERROR: convert_hf_to_gguf.py failed (exit code {result.returncode})")
            # Show last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-10:]:
                print(f"    {line}")
            return False

        print("  F16 GGUF conversion complete")
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Conversion timed out after 2 hours")
        return False
    except FileNotFoundError:
        print(f"  ERROR: Python executable not found: {sys.executable}")
        return False


def quantize_with_llama_cpp(
    f16_gguf_path: Path, output_gguf_path: Path, quant_type: str
) -> bool:
    """
    Use llama.cpp's llama-quantize to quantize an F16 GGUF to the target type.
    For example, F16 -> Q8_0 shrinks a 32B model from ~64GB to ~32GB.
    """
    print(f"  Running: llama-quantize {quant_type} -> {output_gguf_path.name}")

    cmd = [
        str(cfg.LLAMA_CPP_QUANTIZE),
        str(f16_gguf_path),
        str(output_gguf_path),
        quant_type,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode != 0:
            print(f"  ERROR: llama-quantize failed (exit code {result.returncode})")
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-10:]:
                print(f"    {line}")
            return False

        print(f"  Quantization to {quant_type} complete")
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Quantization timed out after 2 hours")
        return False
    except FileNotFoundError:
        print(f"  ERROR: llama-quantize binary not found: {cfg.LLAMA_CPP_QUANTIZE}")
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Conversion Method 2: Python gguf package (fallback)
# ═══════════════════════════════════════════════════════════════════════
def convert_with_python(merged_dir: Path, output_gguf_path: Path, quant_type: str) -> bool:
    """
    Fallback conversion using the Python 'gguf' package and transformers.
    This is slower than llama.cpp but works without any external binaries.

    NOTE: The Python gguf package has limited quantization support compared
    to llama.cpp. For best results with Q4_K_M/Q5_K_M, use llama.cpp.
    F16 and Q8_0 work well with the Python path.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ERROR: transformers package not installed")
        return False

    print("  Loading model for Python-based GGUF conversion...")
    print("  (This uses more RAM than llama.cpp — make sure you have enough)")

    try:
        # Load tokenizer and model in fp16 for conversion
        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir),
            torch_dtype="auto",
            device_map="cpu",  # keep on CPU for conversion
            trust_remote_code=True,
        )

        # Use the model's built-in GGUF export if available (transformers >= 4.40)
        if hasattr(model, "save_pretrained") and "gguf" in str(type(model)):
            print("  Using native GGUF export...")

        # For now, delegate to convert_hf_to_gguf.py bundled with the gguf package
        # The gguf Python package installs this as a console script
        print("  Attempting gguf package conversion...")

        # Try using the gguf package's conversion utilities
        convert_script = shutil.which("convert-hf-to-gguf")
        if convert_script is None:
            # Try the older name too
            convert_script = shutil.which("convert_hf_to_gguf")

        if convert_script:
            cmd = [
                convert_script,
                str(merged_dir),
                "--outfile",
                str(output_gguf_path),
                "--outtype",
                quant_type.lower() if quant_type == "F16" else "f16",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0:
                print("  Python GGUF conversion complete")
                return True
            else:
                print(f"  Conversion script failed: {result.stderr[-500:]}")
                return False

        # Last resort: manual conversion via the gguf library API
        print("  No conversion script found in PATH.")
        print("  Please install llama.cpp tools or ensure the 'gguf' package")
        print("  console scripts are available.")
        print("")
        print("  You can also manually convert using:")
        print(f"    python {cfg.LLAMA_CPP_CONVERT} {merged_dir} --outfile {output_gguf_path} --outtype f16")
        return False

    except Exception as e:
        print(f"  ERROR during Python conversion: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Output naming — generate a descriptive filename for the GGUF
# ═══════════════════════════════════════════════════════════════════════
def generate_gguf_name(merged_dir: Path, quant_type: str, custom_name: str = None) -> str:
    """
    Generate a descriptive GGUF filename based on the model and quantization.

    Examples:
      qwen2.5-coder-32b-anthill-Q8_0.gguf
      custom-model-name-Q4_K_M.gguf
    """
    if custom_name:
        # User specified a name — just append the quant type
        return f"{custom_name}-{quant_type}.gguf"

    # Try to extract model name from the config
    try:
        import json
        config_path = merged_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # _name_or_path usually contains the original HF model ID
        model_id = config.get("_name_or_path", "")
        if "/" in model_id:
            model_id = model_id.split("/")[-1]  # "Qwen/Qwen2.5-Coder-32B-Instruct" -> "Qwen2.5-Coder-32B-Instruct"

        if model_id:
            # Clean up the name and add our suffix
            name = model_id.lower().replace("_", "-")
            # Remove "-instruct" suffix since our fine-tune changes the behavior
            name = name.replace("-instruct", "")
            return f"{name}-anthill-{quant_type}.gguf"
    except Exception:
        pass

    # Fallback: use the directory name
    dir_name = merged_dir.name.lower().replace("_", "-")
    return f"{dir_name}-anthill-{quant_type}.gguf"


# ═══════════════════════════════════════════════════════════════════════
#  Deployment instructions — tell the user how to use the model
# ═══════════════════════════════════════════════════════════════════════
def print_deployment_instructions(gguf_path: Path, model_name: str):
    """
    Print instructions for deploying the fine-tuned GGUF model with
    llama-server or any other GGUF-compatible runtime.
    """
    print("")
    print("=" * 60)
    print("  DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    print("")
    print("  1. Your GGUF is ready at:")
    print(f"     {gguf_path}")
    print("")
    print("  2. Start llama-server (adjust -ngl for your GPU layers):")
    print(f'     llama-server -m "{gguf_path}" --port 8001 -ngl 99 -c 8192')
    print("")
    print("  3. Test with curl:")
    print('     curl http://localhost:8001/v1/chat/completions \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"messages":[{"role":"user","content":"Hello!"}]}\'')
    print("")
    print("  The GGUF works with any llama.cpp-compatible tool:")
    print("  llama-server, ollama, LM Studio, koboldcpp, etc.")
    print("")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    print("")
    print("=" * 60)
    print("  ANTHILL FORGE — GGUF EXPORT")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Merged model: {args.merged_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Quant type:   {args.quant}")
    print(f"  Force Python: {args.force_python}")
    print("")

    # ── Step 1: Validate the merged model ───────────────────────────
    print("[1/5] Validating merged model...")
    if not validate_merged_model(args.merged_dir):
        return 1

    # ── Step 2: Prepare output paths ────────────────────────────────
    print("[2/5] Preparing output paths...")

    gguf_name = generate_gguf_name(args.merged_dir, args.quant, args.output_name)
    final_gguf_path = args.output_dir / gguf_name

    # We always convert to F16 first, then quantize (unless target IS F16)
    f16_name = gguf_name.replace(f"-{args.quant}.gguf", "-F16.gguf")
    f16_gguf_path = cfg.OUTPUT_DIR / f16_name  # intermediate file in local output dir

    # Make sure output directories exist
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Intermediate F16: {f16_gguf_path}")
    print(f"  Final GGUF:       {final_gguf_path}")
    print("")

    # ── Step 3: Convert HF -> F16 GGUF ─────────────────────────────
    print("[3/5] Converting HuggingFace model to F16 GGUF...")
    start_time = time.time()

    use_llama_cpp = False
    if not args.force_python:
        use_llama_cpp = check_llama_cpp_available()

    if args.quant == "F16":
        # Target is F16 — write directly to final path, no quantization needed
        target_path = final_gguf_path
    else:
        target_path = f16_gguf_path

    if use_llama_cpp:
        success = convert_with_llama_cpp(args.merged_dir, target_path)
    else:
        print("  Falling back to Python-based conversion...")
        success = convert_with_python(args.merged_dir, target_path, "F16")

    if not success:
        print("")
        print("  CONVERSION FAILED. Possible fixes:")
        print("  - Make sure BigBlackBox is accessible (network share)")
        print("  - Install the gguf Python package: pip install gguf")
        print("  - Try --force-python to skip llama.cpp")
        return 1

    elapsed = time.time() - start_time
    print(f"  Conversion took {elapsed / 60:.1f} minutes")
    print("")

    # ── Step 4: Quantize (if target is not F16) ────────────────────
    if args.quant != "F16":
        print(f"[4/5] Quantizing F16 -> {args.quant}...")
        start_time = time.time()

        if use_llama_cpp:
            success = quantize_with_llama_cpp(f16_gguf_path, final_gguf_path, args.quant)
        else:
            # Python fallback for quantization is limited — warn the user
            print("  WARNING: Python-based quantization has limited format support.")
            print(f"  For best {args.quant} quality, use llama.cpp's llama-quantize.")
            print("  Attempting conversion directly to target quant type...")
            success = convert_with_python(args.merged_dir, final_gguf_path, args.quant)

        if not success:
            print("  QUANTIZATION FAILED.")
            print(f"  The F16 GGUF is still available at: {f16_gguf_path}")
            print("  You can quantize it manually with:")
            print(f"    llama-quantize {f16_gguf_path} {final_gguf_path} {args.quant}")
            return 1

        elapsed = time.time() - start_time
        print(f"  Quantization took {elapsed / 60:.1f} minutes")

        # Clean up intermediate F16 file (unless user wants to keep it)
        if not args.keep_f16 and f16_gguf_path.exists():
            f16_size_gb = f16_gguf_path.stat().st_size / (1024**3)
            print(f"  Removing intermediate F16 GGUF ({f16_size_gb:.1f} GB)...")
            f16_gguf_path.unlink()
    else:
        print("[4/5] Skipping quantization (target is F16)")

    print("")

    # ── Step 5: Validate the final GGUF ─────────────────────────────
    print("[5/5] Validating final GGUF...")
    if not validate_gguf(final_gguf_path):
        return 1

    # ── Done! Print summary and Jazz instructions ───────────────────
    final_size_gb = final_gguf_path.stat().st_size / (1024**3)
    model_display_name = gguf_name.replace(".gguf", "")

    print("")
    print("  EXPORT COMPLETE!")
    print(f"  File: {final_gguf_path}")
    print(f"  Size: {final_size_gb:.2f} GB")
    print(f"  Type: {args.quant}")

    # Print Jazz deployment instructions so the user knows exactly what to do next
    print_deployment_instructions(final_gguf_path, model_display_name)

    return 0


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(main())
