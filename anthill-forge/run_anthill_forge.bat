@echo off
echo ========================================
echo   Anthill Forge - Model Training Pipeline
echo ========================================
echo.

REM ── Show GPU info so the user knows what they're working with ──
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); vram=torch.cuda.get_device_properties(0).total_mem/1024**3 if torch.cuda.is_available() else 0; print('VRAM: %.1f GB' % vram)" 2>nul
echo.

REM ── Menu ───────────────────────────────────────────────────────
echo What would you like to do?
echo.
echo   1) Classic full fine-tune (small models: Phi-2, etc.)
echo   2) QLoRA fine-tune (large models: 14B-32B+)
echo   3) Export trained model to GGUF
echo   4) Full QLoRA pipeline (train + merge + export)
echo   5) Verify setup (check dependencies and GPU)
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto classic
if "%choice%"=="2" goto qlora
if "%choice%"=="3" goto export
if "%choice%"=="4" goto full_pipeline
if "%choice%"=="5" goto verify
echo Invalid choice.
pause
exit /b 1

REM ═══════════════════════════════════════════════════════════════
REM  Option 1: Classic full fine-tune (backward compatible)
REM ═══════════════════════════════════════════════════════════════
:classic
echo.
echo ── Classic Full Fine-Tune ──
echo.
echo This mode uses full fine-tuning for small models (Phi-2, ~2.7B).
echo Requires enough VRAM to hold the entire model in memory.
echo.

REM Check for cleaned datasets (original behavior)
if not exist "..\datasets\processed\pairs_clean.jsonl" (
    echo No cleaned JSONL files found in ..\datasets\processed\
    echo.
    echo Please run preprocessing first:
    echo 1. Use Anthill Spider extension to download conversations
    echo 2. Run anthill-loom\process_latest_chatgpt.bat with the JSON file
    echo.
    pause
    exit /b 1
)

echo Found cleaned dataset: pairs_clean.jsonl
echo.
set /p proceed="Proceed with training? (Y/N): "
if /i not "%proceed%"=="Y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting classic training...
echo.
python train_instruction_model_auto.py

if errorlevel 1 (
    echo Training failed.
    pause
    exit /b 1
)

echo.
echo Training complete!
echo Model saved to: ..\anthill_forge_output\
echo.
pause
exit /b 0

REM ═══════════════════════════════════════════════════════════════
REM  Option 2: QLoRA fine-tune (for large models)
REM ═══════════════════════════════════════════════════════════════
:qlora
echo.
echo ── QLoRA Fine-Tune ──
echo.
echo This mode uses 4-bit quantization + LoRA adapters to fine-tune
echo large models (14B-32B+) on consumer GPUs (12-16GB VRAM).
echo.

REM Let the user override the model via input, or use config default
echo Default model: Qwen/Qwen2.5-Coder-32B-Instruct (from config_qlora.py)
echo Press Enter to use default, or type a HuggingFace model ID:
set /p model_override=""

if "%model_override%"=="" (
    echo Using default model from config.
    python train_qlora.py
) else (
    echo Using model: %model_override%
    python train_qlora.py --model "%model_override%"
)

if errorlevel 1 (
    echo.
    echo QLoRA training failed.
    pause
    exit /b 1
)

echo.
echo QLoRA training complete!
echo Adapters + merged model saved to: C:\anthill_forge_output\qlora\
echo.
echo To export as GGUF, run this script again and choose option 3.
echo.
pause
exit /b 0

REM ═══════════════════════════════════════════════════════════════
REM  Option 3: Export to GGUF
REM ═══════════════════════════════════════════════════════════════
:export
echo.
echo ── GGUF Export ──
echo.
echo Converts a merged HuggingFace model to GGUF format for llama-server.
echo.

REM Let the user pick quantization type
echo Quantization options:
echo   F16    - Full precision (largest, best quality)
echo   Q8_0   - 8-bit (good balance of size and quality)
echo   Q6_K   - 6-bit (smaller, still good quality)
echo   Q5_K_M - 5-bit (smaller, slight quality loss)
echo   Q4_K_M - 4-bit (smallest, more quality loss)
echo.
echo Default: Q8_0 (from config_qlora.py)
set /p quant_type="Enter quant type (or press Enter for default): "

if "%quant_type%"=="" (
    python export_gguf.py
) else (
    python export_gguf.py --quant "%quant_type%"
)

if errorlevel 1 (
    echo.
    echo GGUF export failed.
    pause
    exit /b 1
)

echo.
pause
exit /b 0

REM ═══════════════════════════════════════════════════════════════
REM  Option 4: Full pipeline (QLoRA train + merge + export)
REM ═══════════════════════════════════════════════════════════════
:full_pipeline
echo.
echo ── Full QLoRA Pipeline ──
echo.
echo This will run the complete pipeline:
echo   1. QLoRA fine-tune (train_qlora.py)
echo   2. Export to GGUF (export_gguf.py)
echo.
echo This may take many hours depending on your dataset and model size.
echo.
set /p proceed="Proceed? (Y/N): "
if /i not "%proceed%"=="Y" (
    echo Pipeline cancelled.
    pause
    exit /b 0
)

echo.
echo ── Step 1/2: QLoRA Training ──
echo.
python train_qlora.py

if errorlevel 1 (
    echo.
    echo Training failed. Aborting pipeline.
    pause
    exit /b 1
)

echo.
echo ── Step 2/2: GGUF Export ──
echo.
python export_gguf.py

if errorlevel 1 (
    echo.
    echo GGUF export failed.
    echo Training was successful — you can retry export with option 3.
    pause
    exit /b 1
)

echo.
echo Full pipeline complete!
echo Your fine-tuned GGUF is ready for Jazz.
echo.
pause
exit /b 0

REM ═══════════════════════════════════════════════════════════════
REM  Option 5: Verify setup
REM ═══════════════════════════════════════════════════════════════
:verify
echo.
echo ── Verifying Setup ──
echo.

echo Checking Python packages...
python -c "import torch; print('  torch:', torch.__version__)" 2>nul || echo   torch: NOT INSTALLED
python -c "import transformers; print('  transformers:', transformers.__version__)" 2>nul || echo   transformers: NOT INSTALLED
python -c "import peft; print('  peft:', peft.__version__)" 2>nul || echo   peft: NOT INSTALLED
python -c "import bitsandbytes; print('  bitsandbytes:', bitsandbytes.__version__)" 2>nul || echo   bitsandbytes: NOT INSTALLED
python -c "import accelerate; print('  accelerate:', accelerate.__version__)" 2>nul || echo   accelerate: NOT INSTALLED
python -c "import datasets; print('  datasets:', datasets.__version__)" 2>nul || echo   datasets: NOT INSTALLED
python -c "import gguf; print('  gguf: installed')" 2>nul || echo   gguf: NOT INSTALLED
echo.

echo Checking GPU...
python -c "import torch; print('  CUDA:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); vram=torch.cuda.get_device_properties(0).total_mem/1024**3 if torch.cuda.is_available() else 0; print('  VRAM: %.1f GB' % vram)" 2>nul || echo   Could not detect GPU
echo.

echo Checking llama.cpp tools (BigBlackBox)...
if exist "\\BigBlackBox\a\llama.cpp\convert_hf_to_gguf.py" (
    echo   convert_hf_to_gguf.py: FOUND
) else (
    echo   convert_hf_to_gguf.py: NOT FOUND
)
if exist "\\BigBlackBox\a\llama.cpp\build\bin\llama-quantize.exe" (
    echo   llama-quantize: FOUND
) else (
    echo   llama-quantize: NOT FOUND (will use Python fallback)
)
echo.

echo Checking dataset directory...
if exist "..\datasets\processed\" (
    echo   Dataset dir exists: ..\datasets\processed\
    dir /b "..\datasets\processed\*.jsonl" 2>nul && echo   JSONL files found above
    if not exist "..\datasets\processed\*.jsonl" echo   No JSONL files found yet
) else (
    echo   Dataset dir not found. Run Anthill Loom first.
)
echo.

echo Checking output directories...
if exist "C:\anthill_forge_output\qlora" (
    echo   QLoRA output dir exists: C:\anthill_forge_output\qlora\
) else (
    echo   QLoRA output dir will be created on first run
)
echo.

echo To install missing packages: pip install -r requirements.txt
echo.
pause
exit /b 0
