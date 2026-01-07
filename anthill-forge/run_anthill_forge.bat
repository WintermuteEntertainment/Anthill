@echo off
echo ========================================
echo   Anthill Forge - Model Training Pipeline
echo ========================================
echo.

REM Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.

REM Check for cleaned datasets
if not exist "..\datasets\processed\pairs_clean.jsonl" (
    echo ❌ No cleaned JSONL files found in ..\datasets\processed\
    echo.
    echo Please run preprocessing first:
    echo 1. Use Anthill Spider extension to download conversations
    echo 2. Run anthill-loom\run_chatgpt_pipeline.bat with the JSON file
    echo.
    pause
    exit /b 1
)

echo Found cleaned dataset: pairs_clean.jsonl
echo.

REM Calculate training time estimate
python -c "import os; dataset_path='../datasets/processed/pairs_clean.jsonl'; line_count=0; import torch; if os.path.exists(dataset_path): with open(dataset_path, 'r', encoding='utf-8') as f: line_count=sum(1 for _ in f); print('Dataset size:', line_count, 'training examples'); if torch.cuda.is_available(): hours=line_count * 3 / 2000; print('Estimated training time on GPU: ~%.1f hours' % hours); else: hours=line_count * 3 / 100; print('Estimated training time on CPU: ~%.1f hours (VERY SLOW!)' % hours)"

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