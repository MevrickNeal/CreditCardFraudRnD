@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo   CreditCardFraudRnD - Starting Application
echo ===================================================
echo.

:: Check for venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run 'setup.bat' first.
    pause
    exit /b 1
)

:: Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

:: Start Backend in a separate window
echo [INFO] Starting FastAPI Backend on http://localhost:8000 ...
start "CreditCardFraudRnD Backend" cmd /k "venv\Scripts\activate && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"

:: Wait for a few seconds for backend to initialize
echo [INFO] Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

:: Open Frontend Dashboard
echo [INFO] Opening Dashboard in browser...
start "" "frontend\index.html"

echo.
echo [INFO] Application is running!
echo [INFO] Keep the Backend terminal window open.
echo.
echo Press any key to exit this script (this will NOT stop the backend).
pause >nul
