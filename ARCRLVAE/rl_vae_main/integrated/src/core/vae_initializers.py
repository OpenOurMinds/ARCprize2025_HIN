import torch
import torch.nn as nn
from src.core.vae_layers import ResNetLayer, AttentionLayer
import logging
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    The encoder part of the VAE. It takes the one-hot encoded grid as input
    and outputs the mean and log-variance of the latent distribution.
    """
    def __init__(self, in_channels, latent_dim, hidden_dims, grid_size):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        current_channels = in_channels
        h, w = grid_size
        
        # Build the convolutional blocks for the encoder
        for h_dim in hidden_dims:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                    ResNetLayer(h_dim, h_dim),
                    AttentionLayer(h_dim)
                )
            )
            current_channels = h_dim
            h, w = h // 2, w // 2 # Downsample grid size

        # Final linear layers to get the mean and log-variance
        self.flattened_size = current_channels * h * w
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1) # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    The decoder part of the VAE. It takes a sample from the latent space
    and reconstructs the original one-hot encoded grid.
    """
    def __init__(self, latent_dim, out_channels, hidden_dims, grid_size):
        super().__init__()
        
        h, w = grid_size
        
        # Calculate the size of the initial feature map after encoder
        final_h = h // (2 ** len(hidden_dims))
        final_w = w // (2 ** len(hidden_dims))
        
        # Simple decoder: linear -> reshape -> conv transpose layers
        self.fc = nn.Linear(latent_dim, hidden_dims[-1] * final_h * final_w)
        self.unflatten_shape = (hidden_dims[-1], final_h, final_w)

        # Build decoder layers that mirror the encoder
        self.decoder_layers = nn.ModuleList()
        current_channels = hidden_dims[-1]
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            h_dim = hidden_dims[i-1]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            current_channels = h_dim

        # Final layer to get output channels - use adaptive pooling to ensure correct size
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((h, w))
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.unflatten_shape)

        for layer in self.decoder_layers:
            x = layer(x)
        
        x = self.final_conv(x)
        x = self.adaptive_pool(x)  # Ensure correct output size
        x = self.final_activation(x)
        return x

class ARC_VAE(nn.Module):
    """
    The complete VAE model, combining the encoder and decoder.
    """
    def __init__(self, in_channels, latent_dim, hidden_dims, grid_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims, grid_size)
        self.decoder = Decoder(latent_dim, in_channels, hidden_dims, grid_size)

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

class VAEInitializer:
    """
    A class to handle the initialization and configuration of the VAE model.
    """
    def __init__(self, config):
        self.config = config
        self.in_channels = self.config.get('num_colors')
        self.latent_dim = self.config.get('latent_dim')
        self.hidden_dims = self.config.get('hidden_dims')
        self.grid_size = self.config.get('grid_size')
        
    def initialize_model(self):
        """
        Initializes the VAE model with the given configuration.
        
        Returns:
            ARC_VAE: The initialized VAE model.
        """
        if not all([self.in_channels, self.latent_dim, self.hidden_dims, self.grid_size]):
            logger.error("Missing required configuration parameters for VAE initialization.")
            return None
            
        logger.info("Initializing VAE model with provided configuration.")
        model = ARC_VAE(
            in_channels=self.in_channels,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            grid_size=self.grid_size
        )
        return model

if __name__ == '__main__':
    # Example usage
    mock_config = {
        'num_colors': 10,
        'latent_dim': 128,
        'hidden_dims': [32, 64, 128],
        'grid_size': (30, 30) # Example grid size
    }
    
    initializer = VAEInitializer(mock_config)
    vae_model = initializer.initialize_model()
    
    if vae_model:
        print("VAE Model Initialized Successfully.")
        print(vae_model)
        
        # Check model output shape
        dummy_input = torch.randn(1, 10, 30, 30) # Batch size 1, 10 colors, 30x30 grid
        reconstruction, mu, logvar = vae_model(dummy_input)
        
        print(f"\nReconstruction Tensor Shape: {reconstruction.shape}")
        print(f"Mu Tensor Shape: {mu.shape}")
        print(f"Logvar Tensor Shape: {logvar.shape}")