@echo off
setlocal
title Friday Fraud - Real-Time Sentinel
echo ============================================================
echo   Friday Fraud - Fintech GenAI Fraud Detection Engine
echo   Production Inference Server
echo ============================================================
echo.

:: 1. Self-Diagnostic: Ensure models are actually trained
echo [1/4] Checking for neural network artifacts...
if not exist "models\artifacts\xgb_ensemble.pkl" (
    echo [ERROR] No trained models found in models/artifacts/
    echo Please run 'setup.bat' first to initialize the engine.
    pause
    exit /b 1
)
echo [OK] All models detected.

:: 2. Port Management: Clean up port 8000
echo [2/4] Clearing port 8000 (killing ghost processes)...
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr "0.0.0.0:8000"') DO (
    taskkill /pid %%T /f >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: 3. Boot Server: Start FastAPI
echo [3/4] Starting AI Engine (FastAPI + PyTorch)...
cd /d "%~dp0backend"
start "FRIDAY_FRAUD_BACKEND" /min python -m uvicorn app:app --port 8000
timeout /t 3 /nobreak >nul

:: 4. Launch UX: Open Browser
echo [4/4] Handshake complete. Launching dashboard...
echo.
echo ============================================================
echo   SERVER ACTIVE AT: http://localhost:8000
echo   Inference Engine: VAE + XGBoost Hybrid
echo ============================================================
echo.
start http://localhost:8000/

exit
