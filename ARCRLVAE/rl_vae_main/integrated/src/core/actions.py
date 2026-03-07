import logging
import numpy as np

logger = logging.getLogger(__name__)

class Actions:
    """
    A class to manage and apply actions for the RL agent.
    
    This class defines a comprehensive set of actions including:
    - Moving a "cursor" (or target) on the grid.
    - Painting a cell with a specific color.
    - Applying global transformations (e.g., flip, rotate, fill).
    """

    def __init__(self, max_grid_size=(30, 30), num_colors=10):
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.action_space_size = self.calculate_action_space_size()

    def calculate_action_space_size(self):
        """
        Calculates the total number of actions in the action space.
        
        The action space is comprised of:
        1. Cursor movements (up, down, left, right): 4 actions
        2. Painting actions (one for each color): `num_colors` actions
        3. Global transformations (flip_h, flip_v, rotate_90): 3 actions
        4. No-op (no operation): 1 action
        
        Total = 4 + num_colors + 3 + 1
        """
        return 4 + self.num_colors + 3 + 1

    def apply_action(self, action_id, grid, cursor_pos):
        """
        Applies a specific action to the grid and cursor position.

        Args:
            action_id (int): The ID of the action to apply.
            grid (np.ndarray): The current grid state.
            cursor_pos (tuple): The current (row, col) cursor position.
        
        Returns:
            tuple: A tuple containing the new grid and new cursor position.
        """
        h, w = grid.shape
        new_grid = np.copy(grid)
        new_cursor_pos = cursor_pos

        # Define action IDs for clarity
        MOVE_UP = 0
        MOVE_DOWN = 1
        MOVE_LEFT = 2
        MOVE_RIGHT = 3
        
        # Color actions start after movement actions
        PAINT_START_ID = 4
        
        # Global transformations start after color actions
        FLIP_H = PAINT_START_ID + self.num_colors
        FLIP_V = FLIP_H + 1
        ROTATE_90 = FLIP_V + 1
        NO_OP = ROTATE_90 + 1

        if action_id == NO_OP:
            # Do nothing
            pass
        elif action_id == MOVE_UP:
            new_cursor_pos = (max(0, cursor_pos[0] - 1), cursor_pos[1])
        elif action_id == MOVE_DOWN:
            new_cursor_pos = (min(h - 1, cursor_pos[0] + 1), cursor_pos[1])
        elif action_id == MOVE_LEFT:
            new_cursor_pos = (cursor_pos[0], max(0, cursor_pos[1] - 1))
        elif action_id == MOVE_RIGHT:
            new_cursor_pos = (cursor_pos[0], min(w - 1, cursor_pos[1] + 1))
        elif action_id >= PAINT_START_ID and action_id < FLIP_H:
            color = action_id - PAINT_START_ID
            r, c = new_cursor_pos
            new_grid[r, c] = color
        elif action_id == FLIP_H:
            new_grid = np.flip(new_grid, axis=1)
        elif action_id == FLIP_V:
            new_grid = np.flip(new_grid, axis=0)
        elif action_id == ROTATE_90:
            new_grid = np.rot90(new_grid)
        else:
            logger.warning(f"Unknown action_id {action_id}, returning copy")

        return new_grid, new_cursor_pos
