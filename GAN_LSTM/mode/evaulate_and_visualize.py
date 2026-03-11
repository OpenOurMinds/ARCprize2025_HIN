import torch
import numpy as np
import json
import sys
import os

# Assuming all your modules are in the current directory
try:
    from preprocessing import Task, preprocess_tasks
    from arc_compressor import ARCCompressor
    from solution_selection import Logger
    from train import take_step
    import visualization
    import initializers
    import layers
    import multitensor_systems
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the required .py files (e.g., preprocessing.py, arc_compressor.py) are in the same directory.")
    sys.exit()

def mock_problem():
    """
    Creates a simple mock problem object to test the model on a new task.
    This simulates a new test task that the model has not seen before.
    """
    return {
        'train': [
            {
                'input': [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                'output': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            }
        ],
        'test': [
            {
                'input': [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
            }
        ]
    }

def main():
    """
    Main function to load the trained models and evaluate them on a new task.
    """
    # Set up CUDA device if available
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_default_device('cpu')
        print("CUDA not available, using CPU.")
    
    # --- 1. Load the trained models ---
    # NOTE: Ensure 'generator.pth' and 'discriminator.pth' are in the same directory.
    print("Loading trained model weights...")
    try:
        generator_weights = torch.load('generator.pth')
    except FileNotFoundError:
        print("Error: 'generator.pth' not found. Please place the trained model file in the current directory.")
        return
    
    # --- 2. Prepare the new task for evaluation ---
    print("\nPreparing a mock task for evaluation...")
    task_name = "evaluation_task"
    problem = mock_problem()
    task = Task(task_name, problem, None)

    # --- 3. Initialize the generator and load the weights ---
    generator = ARCCompressor(task)
    # The weights from the .pth file are a list of tensors, so we load them directly.
    generator.weights_list = generator_weights
    
    # The discriminator (Logger) doesn't have trainable weights in this framework,
    # so we don't need to load anything for it.
    discriminator = Logger(task)
    
    print("Model initialized. Starting solution generation...")

    # --- 4. Run the generator to produce solutions ---
    # We will run the generator for 50 steps to produce a variety of solutions.
    # The discriminator will implicitly select the best ones.
    n_iterations = 50
    for i in range(n_iterations):
        # We don't use an optimizer here as we are not training.
        # We just need to run the forward pass and let the logger do its job.
        take_step(task, generator, None, i, discriminator, is_test_run=True)
    
    print("\nSolution generation complete. Visualizing results...")

    # --- 5. Visualize the final selected solutions ---
    # This will create a plot file in the 'plots' directory.
    os.makedirs('plots', exist_ok=True)
    visualization.plot_solution(discriminator)
    print("Visualization saved to 'plots' directory.")
    print("The plot shows the top 2 solutions selected by the discriminator for the test example.")

if __name__ == "__main__":
    main()