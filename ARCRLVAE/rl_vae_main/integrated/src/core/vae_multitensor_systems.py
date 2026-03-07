import json
import numpy as np
import torch
from src.core.vae_preprocessing import VAEPreprocessor
import logging
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class IOHandler:
    """
    Handles loading and parsing of raw data from a single ARC problem JSON file.
    """
    def __init__(self, config):
        self.config = config

    def load_problem_data(self, json_path):
        """
        Loads an ARC problem from a JSON file.

        Args:
            json_path (str): The file path to the ARC problem JSON.
        
        Returns:
            dict: The parsed JSON data.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded problem from {json_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Error: JSON file not found at {json_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error: Could not decode JSON from {json_path}")
            return None

class ProblemData:
    """
    A container for a single ARC problem's data, including training and test pairs.
    It manages the preprocessing of this data into PyTorch tensors.
    """
    def __init__(self, problem_json, config):
        self.problem_id = None
        self.train_pairs = []
        self.test_pairs = []
        self.config = config
        self.preprocessor = VAEPreprocessor(config)
        self._parse_and_preprocess(problem_json)

    def _parse_and_preprocess(self, problem_json):
        """
        Parses the raw JSON data and preprocesses the grids.
        """
        if problem_json is None:
            return

        self.problem_id = problem_json.get("id", "unknown_id")
        
        # Process training pairs
        for pair in problem_json.get("train", []):
            input_grid = np.array(pair.get("input"))
            output_grid = np.array(pair.get("output"))
            
            # Preprocess to tensors
            preprocessed_input = self.preprocessor.preprocess_grid(input_grid)
            preprocessed_output = self.preprocessor.preprocess_grid(output_grid)
            
            if preprocessed_input is not None and preprocessed_output is not None:
                self.train_pairs.append({
                    "input": preprocessed_input,
                    "output": preprocessed_output,
                    "raw_input": input_grid,
                    "raw_output": output_grid
                })

        # Process testing pairs
        for pair in problem_json.get("test", []):
            input_grid = np.array(pair.get("input"))
            output_grid = np.array(pair.get("output"))
            
            preprocessed_input = self.preprocessor.preprocess_grid(input_grid)
            preprocessed_output = self.preprocessor.preprocess_grid(output_grid)

            if preprocessed_input is not None and preprocessed_output is not None:
                self.test_pairs.append({
                    "input": preprocessed_input,
                    "output": preprocessed_output,
                    "raw_input": input_grid,
                    "raw_output": output_grid
                })

    def get_data(self):
        """
        Returns the preprocessed training and test data.
        """
        return self.train_pairs, self.test_pairs

if __name__ == '__main__':
    # Example Usage: Simulate a mock ARC problem JSON
    mock_config = {
        'grid_size': (3, 3),
        'num_colors': 10
    }
    
    mock_json = {
        "train": [
            {
                "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "output": [[2, 2, 2], [1, 1, 1], [0, 0, 0]]
            }
        ]
    }
    
    # Normally, you would load this from a file, but for this example, we use the mock dict.
    # io_handler = IOHandler(mock_config)
    # mock_json = io_handler.load_problem_data("path/to/your/problem.json")
    
    if mock_json:
        problem_data = ProblemData(mock_json, mock_config)
        train_pairs, test_pairs = problem_data.get_data()
        
        print(f"Problem ID: {problem_data.problem_id}")
        
        print("\n--- Training Data ---")
        for i, pair in enumerate(train_pairs):
            print(f"Train Pair {i}:")
            print(f"  Input Tensor Shape: {pair['input'].shape}")
            print(f"  Output Tensor Shape: {pair['output'].shape}")
            print(f"  Raw Input Grid:\n{pair['raw_input']}")
            print(f"  Raw Output Grid:\n{pair['raw_output']}")
            
        print("\n--- Test Data ---")
        for i, pair in enumerate(test_pairs):
            print(f"Test Pair {i}:")
            print(f"  Input Tensor Shape: {pair['input'].shape}")
            print(f"  Output Tensor Shape: {pair['output'].shape}")
            print(f"  Raw Input Grid:\n{pair['raw_input']}")
            print(f"  Raw Output Grid:\n{pair['raw_output']}")