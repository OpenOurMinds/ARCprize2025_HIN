#data_loader
"""
Placeholder for the data loading utilities.

This module will contain functions to:
1. Load ARC JSON datasets.
2. Load the specific prepared PyTorch datasets for the RL model.
3. Standardize the format of loaded data for use across the pipeline.
"""

# The content of this file will be created by merging and refactoring logic from both baseline data loaders.
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
import logging
import numpy as np
from pathlib import Path

# Assuming logger_setup is in the same directory or on the Python path
from src.utils.logger_setup import setup_logging

# Set up logging for this module
setup_logging()
logger = logging.getLogger(__name__)

class ARCDataset(Dataset):
    """
    A custom PyTorch Dataset for the ARC (Abstraction and Reasoning Corpus) tasks.

    This dataset handles the loading and preprocessing of ARC tasks,
    which consist of input/output grid pairs.
    """
    def __init__(self, data: List[Dict[str, Any]], grid_size: Tuple[int, int]):
        """
        Args:
            data (List[Dict[str, Any]]): A list of ARC task dictionaries.
            grid_size (Tuple[int, int]): The target grid size (height, width) for filtering.
        """
        # Parse ARC format: list of tasks, each with train/test pairs
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for task in data:
            for pair in task.get('train', []):
                inp = np.array(pair.get('input'), dtype=np.int64)
                out = np.array(pair.get('output'), dtype=np.int64)
                if inp.shape == grid_size and out.shape == grid_size:
                    pairs.append((inp, out))

        if not pairs:
            logger.warning(f"No train pairs found for grid size: {grid_size}.")

        self.pairs = pairs
        self.grid_size = grid_size
        logger.info(f"Loaded {len(self.pairs)} train pairs for grid size {self.grid_size}.")

    def __len__(self) -> int:
        """Returns the number of tasks in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single task (input/output pair) and converts it to tensors.
        
        Args:
            idx (int): The index of the task to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and
                                               output grids as PyTorch tensors.
        """
        inp, out = self.pairs[idx]
        input_tensor = torch.from_numpy(inp)
        output_tensor = torch.from_numpy(out)
        return input_tensor, output_tensor

def get_data_loader(
    data_path: str,
    grid_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Creates and returns a DataLoader for the specified ARC data.

    Args:
        data_path (str): The file path to the JSON dataset.
        grid_size (Tuple[int, int]): The target grid size to filter by.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    # Load the JSON data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {data_path}: {e}")
        return None

    # Handle both list format and dict format (task_id -> task_data)
    if isinstance(data, dict):
        # Convert dict format to list of tasks
        task_list = []
        for task_id, task_data in data.items():
            task_data['task_id'] = task_id  # Add task_id to task data
            task_list.append(task_data)
        data = task_list

    # Create the dataset and DataLoader
    dataset = ARCDataset(data, grid_size)
    if len(dataset) == 0:
        return None
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return data_loader

def get_vae_data_loader(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_colors: int = 10,
    grid_size: tuple = (30, 30)
) -> DataLoader:
    """
    Creates a DataLoader specifically for the VAE training data.

    This function is designed to handle the VAE's unsupervised learning
    paradigm, where we are only interested in the input grids. It loads all
    grids regardless of size.

    Args:
        data_path (str): The file path to the JSON dataset.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data. Defaults to True.
    
    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {data_path}: {e}")
        return None
    
    grids = []
    
    # Handle both list format and dict format (task_id -> task_data)
    if isinstance(data, dict):
        tasks = list(data.values())
    else:
        tasks = data
    
    for task in tasks:
        # Extract train pairs from each task
        train_pairs = task.get('train', [])
        for pair in train_pairs:
            # The VAE can be trained on both input and output grids to learn
            # the general distribution of "valid" ARC grids.
            if 'input' in pair:
                grids.append(pair['input'])
            if 'output' in pair:
                grids.append(pair['output'])

    # Convert the list of grids to a single tensor
    # NOTE: The grids are of varying sizes. This will not work with a
    # standard tensor. For a real VAE, you would need to pad or resize
    # the grids to a common size. Here, we'll assume a consistent size for the demo.
    # In a real-world scenario, we'd handle this more robustly.
    
    # Pad to target grid size and convert to one-hot encoding
    target_h, target_w = grid_size
    one_hot_grids = []
    for grid in grids:
        grid_np = np.array(grid, dtype=np.int64)
        h, w = grid_np.shape
        
        # Pad or crop to target size
        padded = np.zeros((target_h, target_w), dtype=np.int64)
        start_h = max(0, (target_h - h) // 2)
        start_w = max(0, (target_w - w) // 2)
        end_h = min(target_h, start_h + h)
        end_w = min(target_w, start_w + w)
        padded[start_h:end_h, start_w:end_w] = grid_np[:end_h-start_h, :end_w-start_w]
        
        # Convert to one-hot encoding: (num_colors, height, width)
        one_hot = np.zeros((num_colors, target_h, target_w), dtype=np.float32)
        for i in range(target_h):
            for j in range(target_w):
                color_idx = int(padded[i, j])
                if 0 <= color_idx < num_colors:
                    one_hot[color_idx, i, j] = 1.0
        
        one_hot_grids.append(one_hot)

    # Convert to a tensor
    if not one_hot_grids:
        logger.warning("No grids to load for the VAE. Returning None.")
        return None
    
    data_tensor = torch.tensor(np.stack(one_hot_grids), dtype=torch.float32)

    # Create a simple TensorDataset
    dataset = torch.utils.data.TensorDataset(data_tensor)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return data_loader


# --- Demonstration of Data Loader Usage ---

if __name__ == '__main__':
    # Create a dummy JSON file for demonstration
    dummy_data = [
        {"task_id": "001", "input": [[1,2], [3,4]], "output": [[1,2], [3,4]]},
        {"task_id": "002", "input": [[5,6], [7,8]], "output": [[5,6], [7,8]]},
        {"task_id": "003", "input": [[9,0], [1,2]], "output": [[9,0], [1,2]]},
        # This one has a different grid size and should be filtered out
        {"task_id": "004", "input": [[1,1,1], [1,1,1]], "output": [[2,2,2], [2,2,2]]}
    ]
    dummy_file_path = "dummy_data.json"
    with open(dummy_file_path, 'w') as f:
        json.dump(dummy_data, f)

    # Example 1: Load data for a specific grid size (2x2)
    logger.info("--- Testing standard data loader for 2x2 grids ---")
    data_loader_2x2 = get_data_loader(
        data_path=dummy_file_path,
        grid_size=(2, 2),
        batch_size=2
    )
    
    if data_loader_2x2:
        for inputs, outputs in data_loader_2x2:
            print("Batch of inputs:")
            print(inputs)
            print("Batch of outputs:")
            print(outputs)
            
    # Example 2: Load data for the VAE (no grid size filtering, assumes padding)
    # The VAE loader is designed to load all grids and pad them, as it
    # learns from a diverse set of grid structures.
    logger.info("\n--- Testing VAE data loader (loads all data) ---")
    # To run this correctly, you'd need a more robust dataset.
    # We will use the `get_vae_data_loader` with a modified dummy file for a better demo.
    
    dummy_vae_data = [
        {"input": [[1, 2, 3], [4, 5, 6]]},
        {"input": [[1, 2], [3, 4], [5, 6]]},
    ]
    dummy_vae_file_path = "dummy_vae_data.json"
    with open(dummy_vae_file_path, 'w') as f:
        json.dump(dummy_vae_data, f)
        
    vae_loader = get_vae_data_loader(
        data_path=dummy_vae_file_path,
        batch_size=2
    )

    if vae_loader:
        for batch in vae_loader:
            print("Batch from VAE loader (padded):")
            # The VAE loader returns a single tensor, not an input/output pair
            print(batch[0])
            print(f"Batch shape: {batch[0].shape}")
            
    # Clean up the dummy file
    Path(dummy_file_path).unlink()
    Path(dummy_vae_file_path).unlink()
