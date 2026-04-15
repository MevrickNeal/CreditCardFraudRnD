@echo off
echo ============================================================
echo   Friday Fraud - Fintech GenAI Fraud Detection Engine
echo   Setup Script
echo ============================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
echo.

:: Check if pre-trained model artifacts already exist
IF EXIST "models\artifacts\xgb_ensemble.pkl" (
    echo [2/3] Pre-trained model artifacts found. Skipping retraining.
    echo        ^(Models were trained on IEEE-CIS Fraud Detection dataset^)
) ELSE (
    echo [2/3] No pre-trained artifacts found.
    echo        You will need the IEEE-CIS dataset in data\raw\ to retrain.
    echo        Download from: https://kaggle.com/c/ieee-fraud-detection/data
    echo        Then run: python data_pipeline.py ^& python train_engine.py
)

echo.
echo [3/3] Setup complete!
echo.
echo ============================================================
echo   Run 'run.bat' to start the application.
echo ============================================================
echo.
pause
