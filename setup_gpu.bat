@echo off
echo ========================================
echo   Anthill Forge - GPU Setup Script
echo ========================================
echo.
echo This script will install PyTorch with CUDA support
echo for NVIDIA RTX 4070/4070 Ti GPUs.
echo.

REM Check for existing PyTorch
echo Checking current PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo Installing PyTorch with CUDA 12.1...
echo (This may take a few minutes...)
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing additional required packages...
pip install transformers datasets accelerate

echo.
echo Verifying GPU setup...
python -c "
import torch
print('=' * 50)
print('GPU Setup Verification:')
print('=' * 50)
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('ERROR: CUDA not available!')
    print('Check:')
    print('  1. NVIDIA drivers installed')
    print('  2. CUDA toolkit installed')
    print('  3. PyTorch with CUDA support installed')
print('=' * 50)
"

echo.
echo GPU setup complete!
echo.
pause