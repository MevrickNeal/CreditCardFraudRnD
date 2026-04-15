@echo off
cd /d "%~dp0"

echo ==========================================
echo Backing up Friday Fraud Project to GitHub
echo ==========================================

:: Initialize git if not already
if not exist ".git" (
    git init
)

:: Add all files
git add .

:: Commit
git commit -m "Final Friday Fraud Project: Advanced Hybrid VAE-XGBoost Architecture, Full Documentation, and Code"

:: Create or ensure branch is main
git branch -M main

:: Add the remote origin (suppressing error if already exists)
git remote add origin https://github.com/MevrickNeal/CreditCardFraudRnD.git 2>nul
git remote set-url origin https://github.com/MevrickNeal/CreditCardFraudRnD.git

:: Push forcibly
echo Pushing...
git push -u origin main --force

echo ==========================================
echo Successfully backed up everything!
echo ==========================================
pause
