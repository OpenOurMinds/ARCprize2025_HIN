#vae_preprocessing.py

import torch
import numpy as np
import logging
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class VAEPreprocessor:
    """
    Handles all data preprocessing for the VAE.
    This includes one-hot encoding, normalization, and reshaping grids
    for the model's input.
    """
    def __init__(self, config):
        """
        Initializes the preprocessor with the configuration.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.grid_size = tuple(config['grid_size'])
        self.num_colors = config['num_colors']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_grid(self, grid):
        """
        Converts a NumPy grid into a one-hot encoded PyTorch tensor.

        Args:
            grid (np.array): The input grid, represented as a 2D NumPy array.
            
        Returns:
            torch.Tensor: The preprocessed grid as a PyTorch tensor,
                          ready for model input.
        """
        grid = np.array(grid)
        height, width = grid.shape
        
        # Check for invalid colors before one-hot encoding
        if np.any(grid >= self.num_colors) or np.any(grid < 0):
            invalid_colors = np.unique(grid[np.logical_or(grid >= self.num_colors, grid < 0)])
            logger.warning(
                f"Grid contains invalid colors {invalid_colors}. "
                f"Expected colors in range [0, {self.num_colors-1}]."
            )
            return None
            
        # Resize by padding or center-cropping to target size
        target_h, target_w = self.grid_size
        padded = np.full((target_h, target_w), fill_value=0, dtype=np.int64)

        # crop if larger
        start_i = 0
        start_j = 0
        src = grid
        if height > target_h:
            start_i = (height - target_h) // 2
        if width > target_w:
            start_j = (width - target_w) // 2
        src_cropped = src[start_i:min(start_i + target_h, height), start_j:min(start_j + target_w, width)]

        ph, pw = src_cropped.shape
        padded[:ph, :pw] = src_cropped

        # Create an empty tensor for one-hot encoding
        one_hot_grid = np.zeros((self.num_colors, target_h, target_w), dtype=np.float32)

        # One-hot encode the grid
        for i in range(target_h):
            for j in range(target_w):
                color_index = int(padded[i, j])
                one_hot_grid[color_index, i, j] = 1.0

        # Convert to a PyTorch tensor (C, H, W)
        tensor = torch.from_numpy(one_hot_grid).to(self.device)
        return tensor

    def postprocess_grid(self, tensor):
        """
        Converts a one-hot encoded PyTorch tensor back into a NumPy grid.
        
        Args:
            tensor (torch.Tensor): The output tensor from the VAE decoder.
            
        Returns:
            np.array: The reconstructed grid as a 2D NumPy array.
        """
        if tensor is None:
            return None
            
        # Accept (C,H,W) or (1,C,H,W); move to CPU
        if tensor.dim() == 4:
            tensor = tensor[0]
        tensor = tensor.cpu()
        
        target_h, target_w = self.grid_size
        reconstructed_grid = torch.argmax(tensor, dim=0).numpy()
        if reconstructed_grid.shape != (target_h, target_w):
            reconstructed_grid = reconstructed_grid[:target_h, :target_w]
        
        return reconstructed_grid

if __name__ == '__main__':
    # Example usage
    mock_config = {
        'grid_size': (5, 5),
        'num_colors': 10
    }
    
    preprocessor = VAEPreprocessor(mock_config)
    
    # Create a mock input grid
    mock_input_grid = np.random.randint(0, 10, size=(5, 5))
    
    print("Original Grid:")
    print(mock_input_grid)
    
    # Preprocess the grid
    preprocessed_tensor = preprocessor.preprocess_grid(mock_input_grid)
    
    if preprocessed_tensor is not None:
        print("\nPreprocessed Tensor Shape:")
        print(preprocessed_tensor.shape)
        
        # Postprocess the tensor (simulating a VAE output)
        postprocessed_grid = preprocessor.postprocess_grid(preprocessed_tensor)
        print("\nPostprocessed Grid (reconstructed):")
        print(postprocessed_grid)
        
        # Verify if the grids are identical
        are_same = np.array_equal(mock_input_grid, postprocessed_grid)
        print(f"\nReconstructed grid is identical to original: {are_same}")
