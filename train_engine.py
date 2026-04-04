import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib

from models.generative import VAEGATHybrid, Generator, Critic
from models.ensemble import StreamingEnsemble

def train_vae(X_train, epochs=20, batch_size=256, lr=1e-3):
    print(f"Training VAE-GAT Hybrid on {len(X_train)} samples...")
    input_dim = X_train.shape[1]
    vae = VAEGATHybrid(input_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            
            recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"VAE Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}")
            
    return vae

def train_wgan(X_train_fraud, epochs=50, batch_size=64, lr=1e-4):
    print(f"Training WGAN-GP on {len(X_train_fraud)} fraud samples...")
    input_dim = X_train_fraud.shape[1]
    noise_dim = 100
    
    generator = Generator(noise_dim, input_dim)
    critic = Critic(input_dim)
    
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_fraud))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch in dataloader:
            real_data = batch[0]
            current_batch_size = real_data.size(0)
            
            for _ in range(5):
                opt_C.zero_grad()
                noise = torch.randn(current_batch_size, noise_dim)
                labels = torch.ones(current_batch_size, 1)
                
                fake_data = generator(noise, labels)
                
                critic_real = critic(real_data, labels).mean()
                critic_fake = critic(fake_data.detach(), labels).mean()
                
                loss_C = -(critic_real - critic_fake)
                loss_C.backward()
                opt_C.step()
                
            opt_G.zero_grad()
            noise = torch.randn(current_batch_size, noise_dim)
            fake_data = generator(noise, labels)
            loss_G = -critic(fake_data, labels).mean()
            loss_G.backward()
            opt_G.step()
            
        if (epoch + 1) % 10 == 0:
            print(f"WGAN Epoch [{epoch+1}/{epochs}], Loss G: {loss_G.item():.4f}")
            
    return generator

def initialize_ensemble(X_train, y_train):
    print("Initializing River continuous learning ensemble...")
    ensemble = StreamingEnsemble()
    
    for i in range(len(X_train)):
        ensemble.fit_one(X_train[i], y_train[i])
        
    print(f"Ensemble pre-trained on {ensemble.samples_seen} transactions. Current ROC-AUC: {ensemble.get_metric():.4f}")
    return ensemble

def map_sandbox_profiles(X, y):
    """ Save a normal and a fraud profile vector manually to test inference """
    idx_normal = np.where(y == 0)[0][0]
    idx_fraud = np.where(y == 1)[0][0]
    
    db = {
        '4000123456789010': {'features': X[idx_normal].tolist(), 'type': 'Normal'},
        '5000987654321098': {'features': X[idx_fraud].tolist(), 'type': 'Fraud'}
    }
    
    import json
    with open('models/artifacts/sandbox_database.json', 'w') as f:
        json.dump(db, f, indent=4)
    print("Sandbox profiles mapped and saved.")

if __name__ == '__main__':
    print("Loading preprocessed feature arrays...")
    X = np.load('data/X_processed.npy')
    y = np.load('data/y_processed.npy')
    
    os.makedirs('models/artifacts', exist_ok=True)
    
    X_legit = X[y == 0]
    vae_model = train_vae(X_legit, epochs=10)
    torch.save(vae_model.state_dict(), 'models/artifacts/vae.pth')
    
    X_fraud = X[y == 1]
    g_model = train_wgan(X_fraud, epochs=20) 
    torch.save(g_model.state_dict(), 'models/artifacts/generator.pth')
    
    print("Generating synthetic fraud samples using WGAN-GP...")
    g_model.eval()
    num_synthetic = 5000
    with torch.no_grad():
        noise = torch.randn(num_synthetic, 100)
        labels = torch.ones(num_synthetic, 1)
        synthetic_fraud = g_model(noise, labels).numpy()
    
    print(f"Augmenting training dataset with {num_synthetic} synthetic fraud transactions...")
    X_augmented = np.vstack([X, synthetic_fraud])
    y_augmented = np.concatenate([y, np.ones(num_synthetic)])
    
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]
    
    ensemble = initialize_ensemble(X_augmented, y_augmented)
    joblib.dump(ensemble, 'models/artifacts/river_ensemble.pkl')
    
    map_sandbox_profiles(X, y)
    
    print("\nTraining Engine Complete! All intelligent assets saved.")