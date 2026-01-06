@echo off
echo ========================================
echo   Anthill Forge - Model Training Pipeline
echo ========================================
echo.

REM Check GPU availability first
echo Checking GPU availability...
python -c "
import torch
if torch.cuda.is_available():
    print('✅ GPU available:')
    for i in range(torch.cuda.device_count()):
        print(f'  - {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
else:
    print('❌ GPU not available. Training will be VERY slow on CPU.')
    print('   Run setup_gpu.bat to install CUDA-enabled PyTorch.')
"
echo.

REM Check for cleaned datasets
set "CLEANED_FILES="
for /f "delims=" %%i in ('dir /b "..\datasets\processed\*_clean.jsonl" 2^>nul') do (
    set "CLEANED_FILES=%%i"
)

if "%CLEANED_FILES%"=="" (
    echo ❌ No cleaned JSONL files found in ..\datasets\processed\
    echo.
    echo Please run preprocessing first:
    echo 1. Use Anthill Spider extension to download conversations
    echo 2. Run anthill-loom\run_chatgpt_pipeline.bat with the JSON file
    echo.
    pause
    exit /b 1
)

echo Found cleaned dataset(s):
for /f "delims=" %%i in ('dir /b "..\datasets\processed\*_clean.jsonl"') do (
    echo   - %%i
)
echo.

REM Calculate training time estimate
python -c "
import json
import os
dataset_path = '..\\datasets\\processed\\pairs_clean.jsonl'
if os.path.exists(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    print(f'Dataset size: {line_count:,} training examples')
    
    # Time estimates
    import torch
    if torch.cuda.is_available():
        hours = line_count * 3 / 2000  # ~2000 examples/hour on RTX 4070
        print(f'Estimated training time on GPU: {hours:.1f} hours')
    else:
        hours = line_count * 3 / 100  # ~100 examples/hour on CPU
        print(f'Estimated training time on CPU: {hours:.1f} hours (VERY SLOW!)')
        print('Consider installing CUDA-enabled PyTorch for GPU acceleration.')
"

echo.
set /p proceed="Proceed with training? (Y/N): "
if /i not "%proceed%"=="Y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting training...
echo.

python train_instruction_model_auto.py

if errorlevel 1 (
    echo ❌ Training failed
    pause
    exit /b 1
)

echo.
echo ✅ Training complete!
echo Model saved to: ..\anthill_forge_output\
echo.
pause