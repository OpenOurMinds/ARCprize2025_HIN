from typing import Dict, List
import logging

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class ActionTokenizer:
    """
    A tokenizer for mapping discrete actions to their integer representations.
    
    This class provides a consistent mapping between a human-readable action
    name (e.g., 'ROTATE_90') and an integer token that the RL model can
    output. This abstraction is crucial for maintaining a clean interface
    between the model and the environment.
    """
    def __init__(self):
        """
        Initializes the tokenizer with a predefined set of actions.
        
        The actions are simplified for demonstration purposes and would be
        expanded in a full-fledged ARC solver.
        """
        # A simplified mapping of actions to integers
        self.actions: Dict[str, int] = {
            'ROTATE_90': 0,
            'ROTATE_180': 1,
            'ROTATE_270': 2,
            'FLIP_HORIZONTAL': 3,
            'FLIP_VERTICAL': 4,
            'CROP_TO_BOUNDING_BOX': 5,
            'FILL_WITH_COLOR_0': 6,
            'FILL_WITH_COLOR_1': 7,
            'FILL_WITH_COLOR_2': 8,
            'FILL_WITH_COLOR_3': 9,
            'NO_OP': 10 # A no-operation action for when no change is needed
        }
        
        self.action_names: List[str] = sorted(self.actions.keys(), key=self.actions.get)
        self.action_dim = len(self.actions)
        
        logger.info(f"Initialized ActionTokenizer with {self.action_dim} discrete actions.")
        
    def get_action_id(self, action_name: str) -> int:
        """
        Retrieves the integer ID for a given action name.
        
        Args:
            action_name (str): The name of the action.
            
        Returns:
            int: The corresponding integer ID.
            
        Raises:
            KeyError: If the action name is not found.
        """
        return self.actions[action_name]
        
    def get_action_name(self, action_id: int) -> str:
        """
        Retrieves the name for a given action ID.
        
        Args:
            action_id (int): The integer ID of the action.
            
        Returns:
            str: The corresponding action name.
            
        Raises:
            IndexError: If the action ID is out of range.
        """
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        else:
            raise IndexError(f"Action ID {action_id} is out of bounds.")
