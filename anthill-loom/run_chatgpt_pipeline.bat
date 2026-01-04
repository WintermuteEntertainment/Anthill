@echo off
echo =========================================
echo   ChatGPT Dataset Preparation Pipeline
echo =========================================

python prepare_datasets_parallel.py %1 %2

if errorlevel 1 (
  echo.
  echo Pipeline FAILED.
) else (
  echo.
  echo Pipeline COMPLETE.
)

pause
