import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder to learn the distribution of legitimate transactions.
    Provides a reconstruction error score for anomaly detection.
    """
    def __init__(self, input_dim, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class Generator(nn.Module):
    """ WGAN-GP Generator for conditional structural data synthesis """
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
            nn.Linear(256, 1) # Output a score, not a probability
        )
        
    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)
