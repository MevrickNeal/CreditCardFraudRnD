@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo   CreditCardFraudRnD - Reset Sandbox Data
echo ===================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run 'setup.bat' first.
    pause
    exit /b 1
)

echo [WARNING] This will re-generate the sandbox profiles. 
echo Press any key to continue or Ctrl+C to cancel.
pause >nul

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Generating sandbox profiles...
python generate_profiles.py

echo.
echo [SUCCESS] Sandbox profiles have been reset!
echo.
pause
