@echo off
echo =========================================
echo   ChatGPT Dataset Preparation Pipeline
echo =========================================
echo.

REM Get the latest downloaded ChatGPT file
for /f "delims=" %%i in ('dir /b /o-d "C:\Users\twwca\Downloads\chatgpt_conversations_*.json"') do (
    set "LATEST_FILE=%%i"
    goto :found
)

:found
if not defined LATEST_FILE (
    echo No chatgpt_conversations_*.json files found in Downloads!
    pause
    exit /b 1
)

echo Latest file: %LATEST_FILE%
echo.

REM Set paths
set "INPUT_FILE=C:\Users\twwca\Downloads\%LATEST_FILE%"
set "OUTPUT_DIR=..\datasets\processed"

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Generate unique output filenames so we never overwrite previous runs
set RAW_OUT=%OUTPUT_DIR%\pairs.jsonl
set CLEAN_OUT=%OUTPUT_DIR%\pairs_clean.jsonl
set /a N=2
:check_exists
if exist "%RAW_OUT%" (
    set RAW_OUT=%OUTPUT_DIR%\pairs_%N%.jsonl
    set CLEAN_OUT=%OUTPUT_DIR%\pairs_clean_%N%.jsonl
    set /a N+=1
    goto :check_exists
)

echo Processing: %INPUT_FILE%
echo Output: %CLEAN_OUT%
echo.

REM Step 1: Extract pairs
echo Step 1: Extracting prompt/completion pairs...
python prepare_datasets_parallel.py "%INPUT_FILE%" "%RAW_OUT%"

if errorlevel 1 (
    echo.
    echo [ERROR] Extraction failed.
    pause
    exit /b 1
)

REM Step 2: Deduplicate and clean
echo.
echo Step 2: Deduplicating and filtering...
python dedupe_and_filter.py "%RAW_OUT%" "%CLEAN_OUT%"

if errorlevel 1 (
    echo.
    echo [ERROR] Deduplication failed.
    pause
    exit /b 1
)

echo.
echo [OK] Pipeline complete!
echo Raw pairs: %RAW_OUT%
echo Cleaned pairs: %CLEAN_OUT%
echo.
echo Next: Run training in anthill-forge directory.
echo.
pause
