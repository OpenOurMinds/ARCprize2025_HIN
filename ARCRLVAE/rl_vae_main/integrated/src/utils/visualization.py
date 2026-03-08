import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any
import logging

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def plot_grid(grid: Any, title: str = ""):
    """
    Plots a single ARC grid using matplotlib.
    
    Args:
        grid (Any): The grid to plot, can be a list of lists or a NumPy array.
        title (str): The title for the plot.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
        
    plt.imshow(grid, cmap='Pastel2')
    plt.title(title)
    plt.axis('off')
    plt.show()
    logger.info(f"Displayed plot for: {title}")

def plot_grids(grids: List[Any], titles: List[str] = []):
    """
    Plots multiple grids side-by-side.
    
    Args:
        grids (List[Any]): A list of grids to plot.
        titles (List[str]): A list of titles for each grid.
    """
    num_grids = len(grids)
    fig, axes = plt.subplots(1, num_grids, figsize=(num_grids * 4, 4))
    
    if num_grids == 1:
        axes = [axes] # Ensure axes is iterable for a single plot
        
    for i, grid in enumerate(grids):
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        
        axes[i].imshow(grid, cmap='Pastel2')
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
            
    plt.tight_layout()
    plt.show()
    logger.info("Displayed plot for multiple grids.")