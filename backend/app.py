from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import json
from models.generative import VAEGATHybrid
from models.ensemble import StreamingEnsemble

app = FastAPI(title="Fintech Fraud GenAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Model Artifacts...")
vae_model = VAEGATHybrid(432)
vae_model.load_state_dict(torch.load('models/artifacts/vae.pth', weights_only=True))
vae_model.eval()

ensemble_model = joblib.load('models/artifacts/river_ensemble.pkl')
feature_scaler = joblib.load('models/artifacts/scaler.pkl')

# Load VAE calibration thresholds (computed from normal transactions during training)
with open('models/artifacts/vae_threshold.json', 'r') as f:
    vae_stats = json.load(f)
vae_threshold = vae_stats['p95']  # 95th percentile of normal data MSE
print(f"VAE calibration: normal p95={vae_threshold:.4f}, mean={vae_stats['mean']:.4f}")

with open('models/artifacts/sandbox_database.json', 'r') as f:
    sandbox_db = json.load(f)

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

    profile = sandbox_db.get(tx.card_number)
    if not profile:
        raise HTTPException(status_code=404, detail="Card profile not found in sandbox")
        
    features = np.array(profile['features'])
    
    # ------- VAE-GAT Anomaly Score -------
    with torch.no_grad():
        recon_x, _, _ = vae_model(torch.FloatTensor([features]))
        anomaly_score = torch.nn.functional.mse_loss(recon_x, torch.FloatTensor([features])).item()
    
    # Calibrated normalization: scores below p95 of normal data → low risk
    # Scores above p95 scale linearly to 1.0 over a 3x range
    if anomaly_score <= vae_threshold:
        anomaly_normalized = (anomaly_score / vae_threshold) * 0.3  # Below threshold = max 30%
    else:
        excess = (anomaly_score - vae_threshold) / (vae_threshold * 2.0)
        anomaly_normalized = 0.3 + min(excess, 1.0) * 0.7  # Above threshold scales 30%→100%
    
    # ------- Ensemble (SGD) Fraud Probability -------
    # Must scale features with the SAME scaler used during training
    features_scaled = feature_scaler.transform([features])[0]
    fraud_prob = ensemble_model.predict_proba_one(features_scaled)
    
    # ------- Hybrid Fusion -------
    # Weighted: 60% VAE anomaly (unsupervised, primary detector) + 40% ensemble (supervised)
    final_risk = (anomaly_normalized * 0.6) + (float(fraud_prob) * 0.4)
    
    # ------- Real Mathematical XAI using linear coefficients -------
    try:
        coefs = ensemble_model.model.coef_[0]
        contributions = np.abs(coefs * features_scaled)
        top_indices = np.argsort(contributions)[-3:][::-1]
        base_shap = {
            f"Model Feat V_{idx}": float(contributions[idx]) for idx in top_indices
        }
    except Exception:
        base_shap = {"Transaction Vectors": float(fraud_prob) * 0.5}
        
    base_shap["Network Anomaly (VAE-GAT)"] = float(anomaly_normalized)
    
    # Normalize XAI for dashboard visualization bounds
    total_shap = sum(base_shap.values())
    if total_shap > 0:
        base_shap = {k: (v/total_shap) * final_risk for k, v in base_shap.items()}
    
    # Online learning feedback
    y_true = 1 if final_risk > 0.5 else 0
    ensemble_model.fit_one(features_scaled, y_true)
    
    result = {
        "transaction_id": f"TXN-{len(history)+1000}",
        "card_number": tx.card_number[-4:],
        "amount": tx.amount,
        "risk_score": final_risk,
        "vae_anomaly": anomaly_normalized,
        "ensemble_prob": fraud_prob,
        "shap_values": base_shap,
        "status": "Declined" if final_risk > 0.5 else "Approved",
        "system_roc_auc": ensemble_model.get_metric()
    }
    
    history.append(result)
    return result

@app.get("/api/history")
async def get_history():
    return history[-10:]