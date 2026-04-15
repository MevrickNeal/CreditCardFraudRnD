# 🔒 Friday Fraud — Fintech GenAI Fraud Detection Engine

A research-grade **real-time credit card fraud detection system** built with:
- 🧠 **Variational Autoencoder (VAE)** — unsupervised anomaly detection trained on legitimate transactions
- 🎭 **WGAN-GP** — generative model to synthesize realistic fraud patterns for training balance
- 🌲 **XGBoost Hybrid Ensemble** — supervised classifier on VAE latent features with `scale_pos_weight` imbalance correction
- ⚡ **FastAPI** backend + interactive web dashboard

Trained on the [IEEE-CIS Fraud Detection dataset (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection) — 590,000 real anonymized e-commerce transactions.

**Target ROC-AUC: > 0.90**

---

## 🚀 Quick Start (New Computer)

### Prerequisites
- Python 3.9 or higher → [Download](https://python.org)
- Must be added to PATH during installation

### Step 1 — Install Dependencies
Double-click **`setup.bat`** (or run in terminal):
```
setup.bat
```
This installs all required packages. Pre-trained model artifacts are included, so **no retraining needed**.

### Step 2 — Launch the App
Double-click **`run.bat`** (or run in terminal):
```
run.bat
```
The browser will open automatically at `http://localhost:8000`.

---

## 🗂️ Project Structure

```
friday fraud/
├── backend/
│   └── app.py                  # FastAPI server — fraud scoring API
├── frontend/
│   ├── index.html              # Web dashboard UI
│   └── app.js                  # Frontend logic
├── models/
│   ├── generative.py           # VAE + WGAN-GP architecture (PyTorch)
│   ├── ensemble.py             # XGBoost Hybrid Ensemble
│   └── artifacts/              # << Pre-trained model files (included)
│       ├── vae.pth             #    Trained VAE weights
│       ├── generator.pth       #    Trained WGAN-GP weights
│       ├── xgb_ensemble.pkl    #    Trained XGBoost model
│       ├── scaler.pkl          #    Fitted StandardScaler
│       ├── label_encoders.pkl  #    Fitted LabelEncoders
│       └── sandbox_database.json # 5 card profiles mapped to real IEEE-CIS rows
├── data_pipeline.py            # IEEE-CIS data loading, encoding, NaN-filling
├── train_engine.py             # Model training orchestrator
├── requirements.txt            # Python dependencies
├── setup.bat                   # One-click setup script
├── run.bat                     # One-click launch script
└── Definitive_Methodology_Guide.md  # Full academic defense documentation
```

---

## 🃏 Sandbox Card Profiles

The dashboard ships with **5 pre-mapped card profiles** from real IEEE-CIS dataset rows:

| Card Number          | Profile Type        | Expected Result      |
|----------------------|---------------------|----------------------|
| `4000 1234 5678 9010`| Normal Transaction  | ✅ Approved          |
| `5000 9876 5432 1098`| Standard Fraud      | 🔴 Declined (88%+)   |
| `4444 5555 6666 7777`| VPN / IP Anomaly    | 🔴 Declined          |
| `4111 2222 3333 4444`| Synthetic Identity  | 🔴 Declined          |
| `5555 6666 7777 8888`| Velocity Attack     | 🔴 Declined          |

---

## 🏗️ Architecture Overview

```
Raw Transaction → data_pipeline.py
                 (LabelEncode + NaN fill, NO scaling)
                        ↓
               train_engine.py
          (fit StandardScaler ONCE → save scaler.pkl)
                 (fit VAE on legit-only data)
              (fit WGAN-GP on fraud-only data)
        (extract VAE latent features → train XGBoost)
                        ↓
              models/artifacts/*.pkl / *.pth
                        ↓
               backend/app.py (FastAPI)
   (load scaler.pkl → scale raw → VAE → XGBoost → risk score)
                        ↓
              frontend dashboard (browser)
```

**Key Design Decision — Single-Scaler Architecture:**  
`StandardScaler.fit()` is called **once** in `train_engine.py`. The same `scaler.pkl` is reloaded in `backend/app.py` before inference. This prevents the "double-scaling" bug where applying `fit_transform` twice corrupts all feature magnitudes.

---

## 🔬 Retraining (Optional — Requires Dataset)

If you want to retrain from scratch, download the dataset from Kaggle:
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
- Extract to `data/raw/` (needs `train_transaction.csv` and `train_identity.csv`)

Then run:
```powershell
python data_pipeline.py    # Preprocess → saves data/X_processed.npy
python train_engine.py     # Train all models → saves models/artifacts/
```

---

## 📦 Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| `torch`        | VAE + WGAN-GP (PyTorch)          |
| `xgboost`      | Hybrid Ensemble classifier       |
| `fastapi`      | REST API backend                 |
| `uvicorn`      | ASGI server                      |
| `scikit-learn` | StandardScaler, LabelEncoder, AUC|
| `numpy`        | Numerical arrays                 |
| `pandas`       | Data loading/manipulation        |
| `joblib`       | Model serialization              |

---

## 📄 Academic Documentation

See **`Definitive_Methodology_Guide.md`** for full technical defense of all design decisions, including:
- Why VAE over Isolation Forest
- Why WGAN-GP over SMOTE
- Why XGBoost over SGD online-learning
- Root causes of the previous AUC failure (double-scaling + imbalanced stream)
- Mathematical anatomy of all 5 simulated attack vectors
