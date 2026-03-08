import torch
import os
import logging
from typing import Any

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class CheckpointHandler:
    """
    A utility class for saving and loading model and optimizer checkpoints.
    
    This class simplifies the process of persisting model states to disk,
    enabling training to be resumed from a specific point.
    """
    def __init__(self, checkpoint_dir: str):
        """
        Initializes the checkpoint handler.

        Args:
            checkpoint_dir (str): The directory where checkpoints will be stored.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory created at: {self.checkpoint_dir}")

    def save_checkpoint(self, model: Any, model_name: str, optimizer: Any = None):
        """
        Saves the model's state dictionary and optionally the optimizer's state.

        Args:
            model (Any): The PyTorch model to save.
            model_name (str): The name of the model (e.g., 'vae', 'rl_policy').
            optimizer (Any): The optimizer to save. Defaults to None.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.pth')
        
        state = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved for {model_name} at {checkpoint_path}")

    def save_named(self, model: Any, filename: str, optimizer: Any = None):
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'model_state_dict': model.state_dict(),
        }
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, model: Any, model_name: str, device: torch.device, optimizer: Any = None) -> bool:
        """
        Loads the model's state dictionary and optionally the optimizer's state.

        Args:
            model (Any): The PyTorch model to load the state into.
            model_name (str): The name of the model.
            device (torch.device): The device the model will be on.
            optimizer (Any): The optimizer to load the state into. Defaults to None.
        
        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return False
            
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            
            logger.info(f"Checkpoint loaded successfully for {model_name}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {model_name}: {e}")
            return False

    def load_named(self, model: Any, filename: str, device: torch.device, optimizer: Any = None) -> bool:
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return False
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            return False