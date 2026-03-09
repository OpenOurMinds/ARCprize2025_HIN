import json
import logging
from typing import Dict, Any

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    A class to handle loading and managing configurations from a JSON file.
    
    This manager provides a centralized way to access all project settings,
    including hyperparameters, file paths, and model configurations.
    """
    def __init__(self, config_path: str):
        """
        Initializes the ConfigManager and loads the configuration file.
        
        Args:
            config_path (str): The path to the JSON configuration file.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the specified JSON file.
        
        Returns:
            Dict[str, Any]: The loaded configuration dictionary.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Successfully loaded configuration from {self.config_path}")
            return config_data
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}. Using default empty config.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.config_path}: {e}. Using default empty config.")
            return {}

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a setting by its key. Supports nested keys using dot notation.
        
        Args:
            key (str): The key of the setting to retrieve (e.g., 'vae.learning_rate').
            default (Any): The default value to return if the key is not found.
        
        Returns:
            Any: The value of the setting, or the default value if not found.
        """
        keys = key.split('.')
        current_level = self.config
        
        try:
            for k in keys:
                current_level = current_level[k]
            return current_level
        except (KeyError, TypeError):
            logger.warning(f"Setting '{key}' not found in config. Using default value: {default}")
            return default

# --- A placeholder for the utils config manager, which would reference this one. ---