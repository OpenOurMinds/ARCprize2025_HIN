import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional
import logging
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as used in LLaMA models.
    Applies: SiLU(gate(x)) * up(x) -> down()
    """
    def __init__(self, embed_size, dropout_rate=0.0):
        super().__init__()
        intermediate_size = int(2 * (4 * embed_size) / 3)
        intermediate_size = (intermediate_size + 7) // 8 * 8
        
        self.gate_proj = nn.Linear(embed_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        
        if self.dropout:
            intermediate = self.dropout(intermediate)
        
        return self.down_proj(intermediate)

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class RLModel(nn.Module):
    """
    The core RL-based Generator model. It consists of a CNN for feature
    extraction, a Transformer-based body for contextual reasoning, and
    a policy head for action selection.
    """

    def __init__(self, config):
        """
        Initializes the model with the given configuration.

        Args:
            config (dict): The configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration parameters
        self.grid_size = config['grid_size']
        self.num_colors = config['num_colors']
        self.embedding_size = config['embedding_size']
        self.action_space_size = config['action_space_size']
        self.max_generation_steps = config['max_generation_steps']
        
        # Grid Feature Extractor (CNN)
        # This takes the one-hot encoded grid and learns a compressed feature representation.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.num_colors, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.embedding_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Policy Head
        # This takes the flattened feature representation and outputs logits for each action.
        conv_output_size = self.embedding_size * self.grid_size[0] * self.grid_size[1]
        self.policy_head = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space_size)
        )
        
        logger.info(f"RLModel initialized with {conv_output_size} features feeding into the policy head.")
        
    def forward(self, x):
        """
        The forward pass of the model. Takes a batch of grids and
        outputs action logits.

        Args:
            x (torch.Tensor): A batch of input grids, shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Logits for the policy, shape (N, action_space_size).
        """
        # Feature extraction via CNN
        features = self.feature_extractor(x)
        
        # Flatten the features to feed into the linear policy head
        features = features.view(features.size(0), -1)
        
        # Output action logits
        action_logits = self.policy_head(features)
        
        return action_logits

    # Generation and action application are handled by TrajectoryGenerator.


# Removed RLToken and ActionApplier mock classes. Use src.core.actions and TrajectoryGenerator.

if __name__ == '__main__':
    # Example usage with a mock configuration
    mock_config = {
        'grid_size': (5, 5),
        'num_colors': 10,
        'embedding_size': 128,
        'action_space_size': 50,
        'max_generation_steps': 10
    }
    
    # Initialize the model
    model = RLModel(mock_config)
    
    # Create a mock input grid tensor
    # Shape: (batch_size, num_colors, height, width)
    input_tensor = torch.randn(2, 10, 5, 5) 
    
    # Get action logits
    action_logits = model(input_tensor)
    logger.info(f"Output action logits shape: {action_logits.shape}")

    # Generate a final output grid
    generated_grids = model.generate(input_tensor)
    logger.info(f"Generated output grids shape: {generated_grids.shape}")
