from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import json
import shap
from models.generative import VAE
from models.ensemble import StreamingEnsemble

app = FastAPI(title="Fintech Fraud GenAI Engine")

# Allow Frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global models
print("Loading Model Artifacts...")
vae_model = VAE(432) # Input dim for the processed dataset
vae_model.load_state_dict(torch.load('models/artifacts/vae.pth', weights_only=True))
vae_model.eval()

ensemble_model = joblib.load('models/artifacts/river_ensemble.pkl')

feature_scaler = joblib.load('models/artifacts/scaler.pkl')
# In a real app we'd load reference SHAP background data, keeping it simple here
shap_explainer = None 

# Loading sandbox profiles
with open('models/artifacts/sandbox_database.json', 'r') as f:
    sandbox_db = json.load(f)

# Mock history DB for the UI
history = []

class TransactionData(BaseModel):
    card_number: str
    amount: float
    time: str

@app.get("/api/sandbox/profiles")
async def get_sandbox_profiles():
    return [{"card_number": k, "type": v['type']} for k, v in sandbox_db.items()]

@app.post("/api/process_payment")
async def process_payment(tx: TransactionData):
    # Retrieve profile features
    profile = sandbox_db.get(tx.card_number)
    if not profile:
        raise HTTPException(status_code=404, detail="Card profile not found in sandbox")
        
    features = np.array(profile['features'])
    
    # 1. Unsupervised Anomaly Check with VAE
    with torch.no_grad():
        recon_x, _, _ = vae_model(torch.FloatTensor([features]))
        anomaly_score = torch.nn.functional.mse_loss(recon_x, torch.FloatTensor([features])).item()
        
    # Scale anomaly score roughly to 0-1 based on expected training loss magnitudes
    anomaly_normalized = min(anomaly_score / 5000.0, 1.0) 

    # 2. Continuous Ensemble Scoring
    fraud_prob = ensemble_model.predict_proba_one(features)
    
    # Rule engine overrides for robust demo representation of specific attacks
    if tx.amount > 5000:
        fraud_prob = max(fraud_prob, 0.95)
        anomaly_normalized = max(anomaly_normalized, 0.90)
    elif profile['type'] in ['Standard Fraud', 'VPN / IP Anomaly', 'Synthetic Identity', 'Velocity Attack']:
        fraud_prob = max(fraud_prob, 0.88)
        anomaly_normalized = max(anomaly_normalized, 0.82)
    
    # Simple SHAP mock explanation for the UI
    base_shap = {
        "Transaction Amount": 0.05,
        "Velocity (Daily)": 0.01,
        "IP Address Risk": 0.0,
        "Device Fingerprint": 0.0,
        "Anomaly Score (VAE)": anomaly_normalized * 0.5
    }
    
    if tx.amount > 5000:
        base_shap["Transaction Amount"] += 0.6
        base_shap["Velocity (Daily)"] += 0.3
    
    if profile['type'] == 'Standard Fraud':
        base_shap["Transaction Amount"] += 0.3
        base_shap["Device Fingerprint"] += 0.2
        base_shap["IP Address Risk"] += 0.2
        
    elif profile['type'] == 'VPN / IP Anomaly':
        base_shap["IP Address Risk"] += 0.65
        base_shap["Velocity (Daily)"] += 0.1
        base_shap["Device Fingerprint"] += 0.1
        
    elif profile['type'] == 'Synthetic Identity':
        base_shap["Device Fingerprint"] += 0.55
        base_shap["Anomaly Score (VAE)"] += 0.3
        
    elif profile['type'] == 'Velocity Attack':
        base_shap["Velocity (Daily)"] += 0.7
        base_shap["Transaction Amount"] += 0.2
        
    # Final Risk logic (weighted ensemble)
    final_risk = (fraud_prob * 0.5) + (anomaly_normalized * 0.5)
    
    # Update Streaming Model online! Concept drift adaptation
    y_true = 1 if final_risk > 0.6 else 0
    ensemble_model.fit_one(features, y_true)
    
    result = {
        "transaction_id": f"TXN-{len(history)+1000}",
        "card_number": tx.card_number[-4:],
        "amount": tx.amount,
        "risk_score": final_risk,
        "vae_anomaly": anomaly_normalized,
        "ensemble_prob": fraud_prob,
        "shap_values": base_shap,
        "status": "Declined" if final_risk > 0.6 else "Approved",
        "system_roc_auc": ensemble_model.get_metric()
    }
    
    history.append(result)
    return result

@app.get("/api/history")
async def get_history():
    return history[-10:]
