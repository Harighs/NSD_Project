import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as config
# from utils import perceptual_loss


class EncoderDecoder(nn.Module):
    """
    EncoderDecoder is a neural network model for encoding and decoding data.
    
    It consists of an encoder and a decoder with intermediate layers, 
    designed for tasks that require transforming input data into a different output format.

    Attributes:
        encoder (nn.Sequential): The encoder network.
        intermediate (nn.Sequential): Intermediate layers for additional processing.
        decoder (nn.Sequential): The decoder network.
    """

    def __init__(self, input_dim: int, output_dim: tuple):
        """
        Initialize the EncoderDecoder model.

        Args:
            input_dim (int): The dimensionality of the input data.
            output_dim (tuple): The height and width dimensions for the output data.
        """
        super(EncoderDecoder, self).__init__()

        if not isinstance(output_dim, tuple) or len(output_dim) != 2:
            raise ValueError("Output dimension must be a tuple of (height, width).")

        self.input_dim = input_dim
        self.output_height, self.output_width = output_dim

        # Encoder configuration
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        # Intermediate layers for additional processing
        self.intermediate = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Decoder configuration
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * self.output_height * self.output_width),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the EncoderDecoder network.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            Tensor: The output of the network after processing the input tensor.
        """
        x = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x)
        x = x.view(-1, 3, self.output_height, self.output_width)  # Reshape to image dimensions
        return x


# MODEL VERY DEEP VARIATIONAL AUTOENCODER
class Very_Deep_VAE(nn.Module):
    """
    Very_Deep_VAE is a neural network model for encoding and decoding data.
    """
    def __init__(self, latent_dim, config):
        """
        Initialize the Very Deep VAE model.

        Args:
            input_dim (int): The dimensionality of the input data.
            output_dim (tuple): The height and width dimensions for the output data.
        """
        super(Very_Deep_VAE, self).__init__()

        if not isinstance(config.output_dim, tuple) or len(config.output_dim) != 2:
            raise ValueError("Output dimension must be a tuple of (height, width).")

        input_dim = config.input_dim
        output_dim_height , output_dim_width = config.output_dim
        
        # Encoder
        # self.fc1 = nn.Linear(input_dim, input_dim*0.7)
        # self.fc2 = nn.Linear(input_dim*0.7, 64)
        self.fc_mean = nn.Linear(10, latent_dim)
        self.fc_logvar = nn.Linear(10, latent_dim)
        # Decoder
        # self.fc3 = nn.Linear(latent_dim, 64)
        # self.fc4 = nn.Linear(64, 128)
        # self.fc5 = nn.Linear(128, 3*output_dim_height * output_dim_width)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14000),
            nn.BatchNorm1d(14000),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(10000, 7000),
            nn.BatchNorm1d(7000),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(3000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(10, 10)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(10, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1000, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(7000, 10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(14000, 3*output_dim_height * output_dim_width)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return h

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    # def loss_function(outputs, x):
    #     recon_x, mean, logvar = outputs
    #     height, width = config.output_dim
        
    #     loss = perceptual_loss(recon_x.view(-1, 3, height, width), x)
    #     return loss
