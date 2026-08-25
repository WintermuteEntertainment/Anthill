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

REM Generate unique output filenames so we never overwrite previous runs
set RAW_OUT=%OUTPUT_DIR%\pairs.jsonl
set CLEAN_OUT=%OUTPUT_DIR%\pairs_clean.jsonl
set /a N=2
:check_raw
if exist "%RAW_OUT%" (
    set RAW_OUT=%OUTPUT_DIR%\pairs_%N%.jsonl
    set CLEAN_OUT=%OUTPUT_DIR%\pairs_clean_%N%.jsonl
    set /a N+=1
    goto :check_raw
)

echo Input: %INPUT_FILE%
echo Output: %CLEAN_OUT%
echo.

REM Step 1: Extract pairs
echo Step 1: Extracting prompt/completion pairs...
python prepare_datasets_parallel.py "%INPUT_FILE%" "%RAW_OUT%"

if errorlevel 1 (
    echo [ERROR] Extraction failed
    pause
    exit /b 1
)

REM Step 2: Deduplicate and clean
echo.
echo Step 2: Deduplicating and filtering...
python dedupe_and_filter.py "%RAW_OUT%" "%CLEAN_OUT%"

if errorlevel 1 (
    echo [ERROR] Deduplication failed
    pause
    exit /b 1
)

echo.
echo [OK] Preprocessing complete!
echo.
echo Raw pairs: %RAW_OUT%
echo Cleaned pairs: %CLEAN_OUT%
echo.
echo Next: Run training in anthill-forge directory.
echo.

pause
