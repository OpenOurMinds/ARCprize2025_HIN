#clogger_setup
"""
Placeholder for the centralized logging setup.

This module will contain a `setup_logging` function to configure the root logger,
ensuring consistent logging across all modules in the project.
"""

# The content of this file will be created to handle log file and console output configuration.
import logging
import sys
from typing import Optional

def setup_logging(
    logging_level: Optional[int] = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Configures a standardized logging setup for the project.

    This function sets up the root logger to output messages to the console
    with a clear and consistent format.

    Args:
        logging_level (Optional[int]): The desired logging level.
            Defaults to logging.INFO. Use logging.DEBUG for more verbose output.
        log_file (Optional[str]): Path to a file where logs should also be written.
            If None, logs are only sent to the console.
    """
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # Define the log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a StreamHandler to log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional: Add a FileHandler if a log file path is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ensure no duplicate handlers from previous runs
    if len(logger.handlers) > (1 + int(log_file is not None)):
        logger.handlers = [console_handler]
        if log_file:
            logger.handlers.append(file_handler)

# --- Demonstration of Logger Setup ---

if __name__ == '__main__':
    # Set up logging at the INFO level
    setup_logging(logging_level=logging.INFO)
    
    # Get a logger for this specific module
    demo_logger = logging.getLogger(__name__)

    # Example log messages at different levels
    demo_logger.debug("This is a DEBUG message. It will not be shown by default.")
    demo_logger.info("This is an INFO message. It provides general information.")
    demo_logger.warning("This is a WARNING message. It indicates a potential issue.")
    demo_logger.error("This is an ERROR message. It indicates an error has occurred.")
    demo_logger.critical("This is a CRITICAL message. It indicates a serious failure.")

    # You can change the level to see more verbose output
    print("\n--- Changing logging level to DEBUG to show all messages ---")
    setup_logging(logging_level=logging.DEBUG)
    
    # Get the logger again (it now has the new level)
    demo_logger = logging.getLogger(__name__)

    # All messages will now be displayed
    demo_logger.debug("This is a DEBUG message. It is now visible.")
    demo_logger.info("This is an INFO message.")
    demo_logger.warning("This is a WARNING message.")
    demo_logger.error("This is an ERROR message.")
    demo_logger.critical("This is a CRITICAL message.")