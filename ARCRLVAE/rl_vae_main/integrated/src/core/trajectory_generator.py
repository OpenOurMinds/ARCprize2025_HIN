import torch
import numpy as np
import logging
from src.core.actions import Actions
from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class TrajectoryGenerator:
    """
    Generates solution candidates for ARC tasks using a trained RL policy.
    It takes an initial grid and an RL model and produces a sequence of
    actions to transform the grid into a solution.
    """

    def __init__(self, rl_model, config):
        """
        Initializes the generator with a trained RL model and configuration.
        """
        self.model = rl_model
        self.config = config
        self.max_steps = config.get('max_steps', 50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the Actions manager
        self.actions = Actions(
            max_grid_size=tuple(config['grid_size']),
            num_colors=config['num_colors']
        )
        self.action_space_size = self.actions.action_space_size
        self.temperature = config.get('temperature', 1.0)
        self.model.eval()

    def _generate_single_trajectory(self, initial_grid, initial_cursor_pos):
        """
        Generates a single sequence of actions and the resulting grid.
        
        This method uses a greedy or probabilistic sampling approach based on the
        RL model's output to select the next best action at each step.
        """
        current_grid = np.copy(initial_grid)
        current_cursor_pos = initial_cursor_pos
        
        with torch.no_grad():
            for _ in range(self.max_steps):
                # Prepare input tensor
                grid_tensor = torch.from_numpy(current_grid).long().to(self.device).unsqueeze(0).unsqueeze(0)
                cursor_tensor = torch.tensor(current_cursor_pos, dtype=torch.long).to(self.device).unsqueeze(0)
                
                # Model inference
                logits = self.model(grid_tensor, cursor_tensor)
                
                # Apply temperature for sampling
                logits = logits / self.temperature
                
                # Sample an action from the logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Check for NaNs and handle them
                if torch.isnan(probabilities).any():
                    logger.warning("NaNs detected in probabilities. Using uniform sampling.")
                    action_id = torch.randint(0, self.action_space_size, (1,)).item()
                else:
                    action_id = torch.multinomial(probabilities, num_samples=1).item()

                # Apply the action
                # Use the Actions class to apply the action and get the new state
                current_grid, current_cursor_pos = self.actions.apply_action(action_id, current_grid, current_cursor_pos)
                
                # Add an optional stopping condition here if needed,
                # e.g., if the grid hasn't changed for N steps.

        return current_grid

    def generate_candidates(self, initial_grid, num_samples=1, beam_width=1):
        """
        Generates multiple solution candidates using beam search or
        independent rollouts.
        
        Args:
            initial_grid (np.ndarray): The starting grid for the trajectory.
            num_samples (int): The number of independent trajectories to generate.
            beam_width (int): The number of candidates to keep at each step.
                              (Note: This simple implementation currently only
                               supports a beam width of 1, effectively performing
                               greedy search for each sample.)
        
        Returns:
            list: A list of final candidate grids (np.ndarray).
        """
        candidates = []
        initial_cursor_pos = (0, 0) # Start cursor at top-left
        
        for _ in range(num_samples):
            final_grid = self._generate_single_trajectory(initial_grid, initial_cursor_pos)
            candidates.append(final_grid)
            
        return candidates