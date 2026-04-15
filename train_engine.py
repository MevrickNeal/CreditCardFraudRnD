"""
Friday Fraud - Training Engine & Evaluation Pipeline
Updated: 2026-04-15
"""
import os, torch, torch.nn as nn, torch.optim as optim, numpy as np, joblib
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from models.generative import VAE, Generator, Critic
from models.ensemble import HybridEnsemble

# ── Loss ────────────────────────────────────────────────────────────────────
def vae_loss_function(recon_x, x, mu, logvar):
    RECON = F.smooth_l1_loss(recon_x, x, reduction='mean')
    KLD   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON + (0.01 * KLD)

# ── VAE ─────────────────────────────────────────────────────────────────────
def train_vae(X_train, epochs=20, batch_size=256, lr=1e-4):
    print(f"Training VAE on {len(X_train)} samples (Smooth L1 + Beta-KLD)...")
    input_dim = X_train.shape[1]
    vae = VAE(input_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    dataset    = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    return vae

# ── WGAN-GP ─────────────────────────────────────────────────────────────────
def train_wgan(X_train_fraud, epochs=50, batch_size=64, lr=1e-4):
    print(f"Training WGAN-GP on {len(X_train_fraud)} fraud samples...")
    input_dim  = X_train_fraud.shape[1]
    noise_dim  = 100
    generator  = Generator(noise_dim, input_dim)
    critic     = Critic(input_dim)
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_C = optim.Adam(critic.parameters(),    lr=lr, betas=(0.5, 0.9))
    dataset    = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_fraud))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch in dataloader:
            real_data = batch[0]
            bs = real_data.size(0)
            for _ in range(5):
                opt_C.zero_grad()
                noise    = torch.randn(bs, noise_dim)
                labels   = torch.ones(bs, 1)
                fake_data = generator(noise, labels)
                c_real  = critic(real_data, labels).mean()
                c_fake  = critic(fake_data.detach(), labels).mean()
                loss_C  = -(c_real - c_fake)
                loss_C.backward()
                opt_C.step()
            opt_G.zero_grad()
            noise     = torch.randn(bs, noise_dim)
            fake_data = generator(noise, labels)
            loss_G    = -critic(fake_data, labels).mean()
            loss_G.backward()
            opt_G.step()
        if (epoch + 1) % 10 == 0:
            print(f"  WGAN Epoch [{epoch+1}/{epochs}], Loss G: {loss_G.item():.4f}")
    return generator

# ── Hybrid Features ─────────────────────────────────────────────────────────
def extract_hybrid(vae, X_scaled):
    """Returns (64 latent dims + 1 recon error) per sample."""
    vae.eval()
    with torch.no_grad():
        X_t     = torch.FloatTensor(X_scaled)
        recon, mu, _ = vae(X_t)
        recon_err = torch.mean((X_t - recon)**2, dim=1).unsqueeze(1)
        return torch.cat((mu, recon_err), dim=1).numpy()

# ── Ensemble Training ────────────────────────────────────────────────────────
def train_hybrid_ensemble(vae, X_scaled, y_train):
    print("Building XGBoost hybrid features from VAE latent space...")
    X_hybrid = extract_hybrid(vae, X_scaled)
    print(f"Hybrid feature shape: {X_hybrid.shape}  (64 Latent + 1 Error)")
    ensemble = HybridEnsemble()
    ensemble.train(X_hybrid, y_train)
    return ensemble, X_hybrid

# ── Sandbox Profiles ─────────────────────────────────────────────────────────
def map_sandbox_profiles(vae, X_scaled, X_raw, y):
    """Map 5 card profiles to real IEEE-CIS array entries."""
    i_norm  = np.where(y == 0)[0][0]
    i_fr1   = np.where(y == 1)[0][0]
    i_fr2   = np.where(y == 1)[0][min(10,  len(np.where(y==1)[0])-1)]
    i_fr3   = np.where(y == 1)[0][min(20,  len(np.where(y==1)[0])-1)]
    i_norm2 = np.where(y == 0)[0][min(100, len(np.where(y==0)[0])-1)]

    def get_hybrid(idx):
        x_s = X_scaled[[idx]]
        x_r = X_raw[[idx]]
        h   = extract_hybrid(vae, x_s)
        return h[0].tolist(), x_r[0].tolist()

    import json
    db = {}
    for card, idx, ctype in [
        ('4000123456789010', i_norm,  'Normal'),
        ('5000987654321098', i_fr1,   'Standard Fraud'),
        ('4444555566667777', i_fr2,   'VPN / IP Anomaly'),
        ('4111222233334444', i_fr3,   'Synthetic Identity'),
        ('5555666677778888', i_norm2, 'Velocity Attack'),
    ]:
        feats, raw = get_hybrid(idx)
        db[card] = {'features': feats, 'raw': raw, 'type': ctype}

    with open('models/artifacts/sandbox_database.json', 'w') as f:
        json.dump(db, f, indent=4)
    print("Sandbox profiles mapped.")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Friday Fraud — Training Engine ===")
    # 1. Load pre-processed (label-encoded, NaN-filled, NOT yet scaled) data
    X_raw = np.load('data/X_processed.npy')
    y     = np.load('data/y_processed.npy')
    print(f"Loaded: X={X_raw.shape}, fraud rate={y.mean()*100:.2f}%")

    os.makedirs('models/artifacts', exist_ok=True)

    # 2. Single StandardScaler fit here (prevents double-scaling bug)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, 'models/artifacts/scaler.pkl')
    print("Scaler fitted and saved.")

    # 3. Train VAE on legit-only scaled data
    X_legit = X_scaled[y == 0]
    vae_model = train_vae(X_legit, epochs=20)
    torch.save(vae_model.state_dict(), 'models/artifacts/vae.pth')

    # 4. Train WGAN on fraud-only scaled data
    X_fraud = X_scaled[y == 1]
    g_model = train_wgan(X_fraud, epochs=20)
    torch.save(g_model.state_dict(), 'models/artifacts/generator.pth')

    # 5. Train hybrid XGBoost ensemble on full scaled data
    ensemble, _ = train_hybrid_ensemble(vae_model, X_scaled, y)
    joblib.dump(ensemble, 'models/artifacts/xgb_ensemble.pkl')

    # 6. Map sandbox card profiles (pass both scaled + raw for raw storage)
    map_sandbox_profiles(vae_model, X_scaled, X_raw, y)

    print("\n=== Training complete! All artifacts saved. ===")
