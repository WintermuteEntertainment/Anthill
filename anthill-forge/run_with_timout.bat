@echo off
echo ========================================
echo   Anthill Forge - Safe Training Wrapper
echo ========================================
echo.

REM Set maximum training time (in hours)
set MAX_HOURS=12

REM Check if training is already running
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq Anthill*" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL% EQU 0 (
    echo ❌ Training already appears to be running!
    echo.
    tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq Anthill*"
    echo.
    pause
    exit /b 1
)

echo Maximum training time: %MAX_HOURS% hours
echo Starting training with timeout protection...
echo.

REM Start training in a separate window so we can monitor it
start "Anthill Training" cmd /c "python train_instruction_model_auto.py"

echo.
echo Training started in new window.
echo Process will auto-terminate after %MAX_HOURS% hours.
echo.

REM Wait a moment for process to start
timeout /t 5 /nobreak > NUL

REM Find the training process
for /f "tokens=2" %%i in ('tasklist /FI "WINDOWTITLE eq Anthill*" /FO CSV ^| findstr python') do (
    set PID=%%~i
)

if not defined PID (
    echo Could not find training process!
    pause
    exit /b 1
)

echo Training PID: %PID%
echo.

REM Calculate timeout in seconds
set /a TIMEOUT_SECONDS=%MAX_HOURS% * 3600

REM Start monitoring script in background
start "Training Monitor" /MIN cmd /c "python monitor_training.py %PID% training_monitor_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log 60"

echo Monitoring started...
echo.

REM Wait for timeout or completion
echo Waiting for training to complete (max %MAX_HOURS% hours)...
echo Press Ctrl+C to stop training early.
echo.

timeout /t %TIMEOUT_SECONDS% /nobreak > NUL

REM Check if training is still running
tasklist /FI "PID eq %PID%" 2>NUL | find /I "%PID%" >NUL
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ⏰ Timeout reached! Stopping training...
    taskkill /PID %PID% /F
    echo Training stopped.
) else (
    echo.
    echo ✅ Training completed successfully.
)

echo.
echo Monitor log saved to: training_monitor_*.log
echo.
pause