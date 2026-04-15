"""
Friday Fraud - Generative Deep Learning (VAE + WGAN-GP)
Updated: 2026-04-15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder — matches the Kaggle notebook architecture exactly.
    2-layer encoder (input -> 256 -> latent), logvar clamping, Smooth L1 + Beta-KLD loss.
    """
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        logvar = torch.clamp(logvar, min=-20, max=20)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

class Generator(nn.Module):
    """ WGAN-GP Generator for conditional synthetic fraud data generation """
    def __init__(self, noise_dim, output_dim, condition_dim=1):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim)
        )
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

class Critic(nn.Module):
    """ WGAN-GP Critic for scoring data realism """
    def __init__(self, input_dim, condition_dim=1):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)
