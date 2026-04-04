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

db = {
    '4000123456789010': {'features': X[idx_normal].tolist(), 'type': 'Normal'},
    '5000987654321098': {'features': X[idx_fraud].tolist(), 'type': 'Standard Fraud'},
    '4444555566667777': {'features': X[idx_fraud_2].tolist(), 'type': 'VPN / IP Anomaly'},
    '4111222233334444': {'features': X[idx_fraud_3].tolist(), 'type': 'Synthetic Identity'},
    '5555666677778888': {'features': X[idx_normal_2].tolist(), 'type': 'Velocity Attack'}
}

with open('models/artifacts/sandbox_database.json', 'w') as f:
    json.dump(db, f, indent=4)
    
print("Sandbox profiles expanded to include VPN, Synthetic Identity, and Velocity attacks!")