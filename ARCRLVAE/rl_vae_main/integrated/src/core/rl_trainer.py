import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.utils.data import DataLoader
from src.core.rl_model import RLModel
from src.utils.checkpoint_handler import CheckpointHandler
from src.core.trajectory_generator import TrajectoryGenerator

logger = logging.getLogger(__name__)

class Discriminator(nn.Module):
    """
    Mock class for the Discriminator model.
    In a real implementation, this would be a sophisticated neural network
    that learns to distinguish between human-generated and machine-generated grids.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.discriminator = nn.Linear(config['num_colors'] * config['grid_size'][0] * config['grid_size'][1], 1)
        
    def forward(self, x):
        """
        Takes a grid and outputs a scalar score indicating how "real" it is.
        A higher score means the discriminator thinks the grid is more likely
        to be a human-created ground truth.
        """
        batch_size = x.size(0)
        flattened_x = x.view(batch_size, -1)
        return torch.sigmoid(self.discriminator(flattened_x))

class RewardFunction:
    """
    Mock class for the Reward Function.
    This provides a simplified way to calculate a reward for a generated trajectory.
    In a real system, the reward would be tied to the Discriminator's output.
    """
    def __init__(self, discriminator, config):
        self.discriminator = discriminator
        self.config = config

    def calculate_reward(self, final_grid, ground_truth_grid):
        """
        Calculates a reward for the generated grid based on its similarity to the ground truth.
        """
        # A simple reward: a value between 0 and 1 based on pixel-wise similarity.
        # This is a proxy for the discriminator's output.
        final_grid = final_grid.cpu().detach().numpy()
        ground_truth_grid = ground_truth_grid.cpu().detach().numpy()
        
        # Ensure grids are the same size before comparison
        if final_grid.shape != ground_truth_grid.shape:
            return 0.0
            
        similarity = np.mean(final_grid == ground_truth_grid)
        return similarity

class RLTrainer:
    """
    Manages the training loop for the RL model using a DataLoader of tensors.
    """

    def __init__(self, rl_model: RLModel, discriminator: nn.Module, config: dict, checkpoint_dir: str | None = None):
        """
        Initializes the trainer with the models and configuration.

        Args:
            rl_model (RLModel): The Generator model.
            discriminator (Discriminator): The Discriminator model.
            config (dict): The configuration dictionary.
        """
        self.rl_model = rl_model
        self.discriminator = discriminator
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimizers
        self.optimizer = optim.Adam(self.rl_model.parameters(), lr=config['learning_rate'])
        
        # Trajectory Generator
        self.trajectory_generator = TrajectoryGenerator(rl_model, config)
        
        # Reward Function (uses the discriminator's logic)
        self.reward_function = RewardFunction(discriminator, config)
        self.best_metric = float('-inf')
        self.checkpointer = CheckpointHandler(checkpoint_dir) if checkpoint_dir else None
        
    def train(self, data_loader: DataLoader, num_epochs: int = 100, val_loader: DataLoader | None = None):
        """
        Runs the main training loop for the RL model using a DataLoader that
        yields batches of (input_tensor, output_tensor) as integer grids.
        """
        self.rl_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in data_loader:
                # batch contains tensors of integer color ids: (B, H, W)
                inputs_int = batch[0].cpu().numpy()
                outputs_int = batch[1].cpu().numpy()

                batch_loss = 0.0
                for i in range(len(inputs_int)):
                    input_grid = inputs_int[i]
                    output_grid = outputs_int[i]

                    states, actions, final_grid = self.trajectory_generator.generate_trajectory(input_grid)

                    reward = self.reward_function.calculate_reward(
                        self._grid_to_tensor(final_grid), self._grid_to_tensor(output_grid)
                    )

                    self.optimizer.zero_grad()
                    step_log_probs = []
                    for state, action_id in zip(states, actions):
                        state_tensor = self._grid_to_tensor(state)
                        action_logits = self.rl_model(state_tensor)
                        log_probs = F.log_softmax(action_logits, dim=1)
                        step_log_probs.append(log_probs[:, int(action_id)].squeeze(0))

                    policy_loss = -torch.stack(step_log_probs).sum() * float(reward)
                    policy_loss.backward()
                    self.optimizer.step()

                    batch_loss += policy_loss.item()

                total_loss += batch_loss
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            val_reward = None
            if val_loader is not None:
                val_reward = self.evaluate_reward(val_loader, num_batches=10)
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, ValReward: {val_reward:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

            # Save latest and best
            if self.checkpointer:
                self.checkpointer.save_named(self.rl_model, 'Transformer_latest.pt', optimizer=self.optimizer)
                metric = val_reward if val_reward is not None else -avg_loss
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.checkpointer.save_named(self.rl_model, 'Transformer_best.pt', optimizer=self.optimizer)

    def evaluate_reward(self, data_loader: DataLoader, num_batches: int = 10) -> float:
        self.rl_model.eval()
        rewards = []
        with torch.no_grad():
            for bi, batch in enumerate(data_loader):
                if bi >= num_batches:
                    break
                inputs_int = batch[0].cpu().numpy()
                outputs_int = batch[1].cpu().numpy()
                for i in range(len(inputs_int)):
                    input_grid = inputs_int[i]
                    output_grid = outputs_int[i]
                    states, actions, final_grid = self.trajectory_generator.generate_trajectory(input_grid)
                    r = self.reward_function.calculate_reward(
                        self._grid_to_tensor(final_grid), self._grid_to_tensor(output_grid)
                    )
                    rewards.append(float(r))
        self.rl_model.train()
        return float(np.mean(rewards)) if rewards else 0.0
            
    def _grid_to_tensor(self, grid):
        """Converts a NumPy grid into a one-hot encoded PyTorch tensor."""
        grid = np.array(grid)
        height, width = grid.shape
        num_colors = self.config['num_colors']
        one_hot_grid = np.zeros((height, width, num_colors), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                color_index = int(grid[i, j])
                if 0 <= color_index < num_colors:
                    one_hot_grid[i, j, color_index] = 1.0

        tensor = torch.from_numpy(one_hot_grid).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

if __name__ == '__main__':
    # Example usage with a mock configuration
    mock_config = {
        'grid_size': (5, 5),
        'num_colors': 10,
        'embedding_size': 128,
        'action_space_size': 50,
        'learning_rate': 0.001,
        'max_generation_steps': 10
    }
    
    # Initialize mock models
    rl_model = RLModel(mock_config)
    discriminator = Discriminator(mock_config)
    
    # Create mock training data
    mock_input = np.random.randint(0, 10, size=(5, 5))
    mock_output = np.random.randint(0, 10, size=(5, 5))
    
    input_data = [mock_input]
    output_data = [mock_output]

    # Initialize and run the trainer
    trainer = RLTrainer(rl_model, discriminator, mock_config)
    trainer.train(input_data, output_data, num_epochs=5)
