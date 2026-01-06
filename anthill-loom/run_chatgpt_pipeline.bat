@echo off
echo =========================================
echo   Anthill Loom - ChatGPT Preprocessing Pipeline
echo =========================================
echo.

REM Check for input file
if "%1"=="" (
    echo Usage: %0 ^<input_json^>
    echo Example: %0 "C:\Users\twwca\Downloads\chatgpt_conversations_latest.json"
    echo.
    echo Or drag and drop a JSON file onto this batch file.
    pause
    exit /b 1
)

set INPUT_FILE=%1
set OUTPUT_DIR=..\datasets\processed

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Input: %INPUT_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

REM Step 1: Extract pairs
echo Step 1: Extracting prompt/completion pairs...
python prepare_datasets_parallel.py "%INPUT_FILE%" "%OUTPUT_DIR%\pairs.jsonl"

if errorlevel 1 (
    echo ❌ Extraction failed
    pause
    exit /b 1
)

REM Step 2: Deduplicate and clean
echo.
echo Step 2: Deduplicating and filtering...
python dedupe_and_filter.py "%OUTPUT_DIR%\pairs.jsonl" "%OUTPUT_DIR%\pairs_clean.jsonl"

if errorlevel 1 (
    echo ❌ Deduplication failed
    pause
    exit /b 1
)

echo.
echo ✅ Preprocessing complete!
echo.
echo Raw pairs: %OUTPUT_DIR%\pairs.jsonl
echo Cleaned pairs: %OUTPUT_DIR%\pairs_clean.jsonl
echo.
echo Next: Run training in anthill-forge directory.
echo.

REM Optional: Clean up intermediate file (comment out if you want to keep it)
REM del "%OUTPUT_DIR%\pairs.jsonl"

pause