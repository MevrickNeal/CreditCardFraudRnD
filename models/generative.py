import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEGATHybrid(nn.Module):
    """
    Variational Autoencoder + Graph Attention Hybrid.
    Uses dense self-attention over the batch to simulate graph node aggregation
    and learn relational anomalies between transactions.
    """
    def __init__(self, input_dim, latent_dim=64):
        super(VAEGATHybrid, self).__init__()
        
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dense mini-batch GAT using MultiheadAttention (Self-Attention over the batch graph)
        self.gat_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.2, batch_first=True)
        
        self.post_gat = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x):
        h = self.base_encoder(x)
        # Treat batch as a fully connected graph sequence: (1, Batch, Features)
        h_seq = h.unsqueeze(0) 
        gat_out, _ = self.gat_attention(h_seq, h_seq, h_seq)
        h_graph = gat_out.squeeze(0) + h # Residual Connection
        
        h_final = self.post_gat(h_graph)
        return self.fc_mu(h_final), self.fc_logvar(h_final)
        
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
            nn.Linear(256, 1)
        )
        
    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)