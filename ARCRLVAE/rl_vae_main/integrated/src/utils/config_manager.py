#config_manager

# This utility config manager is a simple passthrough to the core one,
# as per the specified file structure.

from src.core.config_manager import ConfigManager
import logging

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Re-export the core ConfigManager for a consistent import path
# if other modules are configured to import from this location.
ConfigManager = ConfigManager

# A simplified function that could be used for simple, single-level configs.
def get_config_setting(config_path: str, key: str, default: Any = None) -> Any:
    """
    A simple utility function to get a single setting from a config file.
    
    Args:
        config_path (str): Path to the config JSON file.
        key (str): The key of the setting to retrieve.
        default (Any): The default value if the key is not found.
        
    Returns:
        Any: The value of the setting or the default.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data.get(key, default)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        return default