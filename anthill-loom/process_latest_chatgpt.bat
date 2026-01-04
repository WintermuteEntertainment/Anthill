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
set "OUTPUT_DIR=X:\Anthill\Anthill\anthill-loom\datasets\processed"
set "OUTPUT_FILE=%OUTPUT_DIR%\pairs_%date:~-4,4%%date:~-7,2%%date:~-10,2%.jsonl"

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Processing: %INPUT_FILE%
echo Output: %OUTPUT_FILE%
echo.

REM Run the Python script
python prepare_datasets_parallel.py "%INPUT_FILE%" "%OUTPUT_FILE%"

if errorlevel 1 (
    echo.
    echo Pipeline FAILED.
) else (
    echo.
    echo Pipeline COMPLETE.
)

echo.
pause