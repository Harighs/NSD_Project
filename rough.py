import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config as config

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        
        input_dim = config.input_dim
        output_dim_height = config.output_dim[0]
        output_dim_width = config.output_dim[1]
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, output_dim_height * output_dim_width)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def loss_function(outputs, x):
        recon_x, mean, logvar = outputs
        # Binary Cross Entropy Loss
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, config.input_dim), reduction='sum')
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD

