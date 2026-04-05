import numpy as np
import json
import os

print("Loading data...")
X = np.load('data/X_processed.npy')
y = np.load('data/y_processed.npy')

idx_normal = np.where(y == 0)[0][0]
idx_fraud = np.where(y == 1)[0][0]
idx_fraud_2 = np.where(y == 1)[0][10]
idx_fraud_3 = np.where(y == 1)[0][20]
idx_normal_2 = np.where(y == 0)[0][100]

# Inject mathematical anomalies so the real AI catches them organically
feat_standard = X[idx_fraud].copy()
feat_standard[5:50] += 15.0  # Strong deviation across early features
feat_standard[150:200] *= -6.0  # Flip sign of network features
feat_standard[400:420] += 20.0  # Spike late features

feat_vpn = X[idx_fraud_2].copy()
feat_vpn[50:100] *= 20.0  # Massive network feature spike
feat_vpn[150:180] -= 15.0

feat_synthetic = X[idx_fraud_3].copy()
feat_synthetic[0:60] += 20.0  # Broad identity anomaly
feat_synthetic[100:170] -= 18.0
feat_synthetic[280:360] *= -10.0

feat_velocity = X[idx_normal_2].copy()
feat_velocity[2] += 60.0  # Massive transaction amount
feat_velocity[5:55] += 18.0  # Rapid-fire pattern signatures
feat_velocity[280:350] += 20.0

db = {
    '4000123456789010': {'features': X[idx_normal].tolist(), 'type': 'Normal'},
    '5000987654321098': {'features': feat_standard.tolist(), 'type': 'Standard Fraud'},
    '4444555566667777': {'features': feat_vpn.tolist(), 'type': 'VPN / IP Anomaly'},
    '4111222233334444': {'features': feat_synthetic.tolist(), 'type': 'Synthetic Identity'},
    '5555666677778888': {'features': feat_velocity.tolist(), 'type': 'Velocity Attack'}
}

with open('models/artifacts/sandbox_database.json', 'w') as f:
    json.dump(db, f, indent=4)
    
print("Sandbox profiles expanded to include VPN, Synthetic Identity, and Velocity attacks!")