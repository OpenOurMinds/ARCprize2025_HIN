import os
import glob
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

"""Loading JSON data:"""

file_path='*/Users/seungwonlee/ARCprize2025_HIN/re-ARC-dataset/tasks'

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

"""Reading files:

### Function to plot input/output pairs of a task
"""

# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(3, 1), dpi=150)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.tick_params(axis='x', color='r', length=0, grid_color='none')

plt.show()

def plot_task(task, task_solutions, i, t, size=2.5, w1=0.9):
    t=list(training_challenges)[i]
    titleSize=16
    num_train = len(task['train'])
    num_test  = len(task['test'])

    wn=num_train+num_test
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task #{i}, {t}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')

    '''train:'''
    for j in range(num_train):
        plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
        plot_one(axs[1, j], j,task, 'train', 'output', w=w1)

    '''test:'''
    for k in range(num_test):
        plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
        task['test'][k]['output'] = task_solutions[k]
        plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)

    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, wn])

    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)

    axs[1, j+1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5) #widthframe
    fig.patch.set_edgecolor('black') #colorframe
    fig.patch.set_facecolor('#444444') #background

    plt.tight_layout()

    print(f'#{i}, {t}') # for fast and convinience search
    plt.show()

def plot_one(ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
    fs=12
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    #ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

    '''Grid:'''
    ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)

    ax.tick_params(axis='both', color='none', length=0)

    '''sub title:'''
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd')

"""
# <div  style="color:white; border:lightgreen solid;  font-weight:bold; font-size:120%; text-align:center;padding:12.0px; background:black">2. GRID TRANSFORMATION</div>"""

import os
import glob
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.notebook import tqdm # Import tqdm

def plot_task(task, task_solutions, i, t, size=2.5, w1=0.9):
    t=list(training_challenges)[i]
    titleSize=16
    num_train = len(task['train'])
    num_test  = len(task['test'])

    wn=num_train+num_test
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task #{i}, {t}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')

    '''train:'''
    for j in range(num_train):
        plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
        plot_one(axs[1, j], j,task, 'train', 'output', w=w1)

    '''test:'''
    for k in range(num_test):
        plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
        task['test'][k]['output'] = task_solutions[k]
        plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)

    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, wn])

    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)

    axs[1, j+1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5) #widthframe
    fig.patch.set_edgecolor('black') #colorframe
    fig.patch.set_facecolor('#444444') #background

    plt.tight_layout()

    print(f'#{i}, {t}') # for fast and convinience search
    plt.show()

def plot_one(ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
    fs=12
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    #ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

    '''Grid:'''
    ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)

    ax.tick_params(axis='both', color='none', length=0)

    '''sub title:'''
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd')

class GridTransformer:
    """
    A class to apply unique geometric and color transformations to grids.
    The transformations include: rotations (0, 90, 180, 270 degrees), mirroring (horizontal, vertical, main diagonal, anti-diagonal),
    and a 45-degree diagonal transformation. All can also be applied to color-inverted grids.
    """
    @staticmethod
    def rotate_90(grid):
        """Rotates a grid 90 degrees clockwise."""
        # Convert to numpy array for easier manipulation
        grid_np = np.array(grid)
        return np.rot90(grid_np, k=-1).tolist() # k=-1 for clockwise

    @staticmethod
    def rotate_180(grid):
        """Rotates a grid 180 degrees."""
        grid_np = np.array(grid)
        return np.rot90(grid_np, k=-2).tolist() # k=-2 for 180 degrees clockwise

    @staticmethod
    def rotate_270(grid):
        """Rotates a grid 270 degrees clockwise."""
        grid_np = np.array(grid)
        return np.rot90(grid_np, k=-3).tolist() # k=-3 for 270 degrees clockwise

    @staticmethod
    def flip_horizontal(grid):
        """Flips a grid horizontally."""
        grid_np = np.array(grid)
        return np.fliplr(grid_np).tolist()

    @staticmethod
    def flip_vertical(grid):
        """Flips a grid vertically."""
        grid_np = np.array(grid)
        return np.flipud(grid_np).tolist()

    @staticmethod
    def flip_main_diagonal(grid):
        """Flips a grid along its main diagonal (transpose)."""
        grid_np = np.array(grid)
        return grid_np.transpose().tolist()

    @staticmethod
    def flip_anti_diagonal(grid):
        """Flips a grid along its anti-diagonal."""
        # Equivalent to rotating 90 degrees clockwise, then flipping horizontally
        grid_np = np.array(grid)
        return np.fliplr(np.rot90(grid_np, k=-1)).tolist()

    @staticmethod
    def invert_colors(grid):
        """Inverts the colors of a grid using a specific ARC-AGI mapping.
            0->5, 1->4, 2->3, 3->2, 4->1, 5->0, 6->9, 7->8, 8->7, 9->6.
        """
        inversion_map = {
            0: 5, 1: 4, 2: 3, 3: 2, 4: 1,
            5: 0, 6: 9, 7: 8, 8: 7, 9: 6
        }
        # Handle cases where input might be an empty list or contains empty rows
        if not grid or not grid[0]:
            return [[]] if grid else []

        # Use .get to handle colors not explicitly in map, keeping them unchanged
        return [[inversion_map.get(cell, cell) for cell in row] for row in grid]

    @staticmethod
    def transform_grid_45_degree_diagonal(grid, fill_value=0):
        """
        Transforms a 2D grid to a 45-degree diagonal pattern based on
        a specific coordinate mapping. The output grid is cropped to
        remove unnecessary fill_value borders.

        Args:
            grid (list of lists): The input 2D grid.
            fill_value (int): The value to fill blank spaces with.

        Returns:
            list of lists: The new, transformed and cropped grid.
        """
        if not grid or not grid[0]:
            return [[]]

        grid_np = np.array(grid)
        height, width = grid_np.shape

        # Calculate the dimensions of the new rotated grid's bounding box.
        # These dimensions ensure all transformed points fit.
        new_height_raw = height + width
        new_width_raw = height + width

        # Create a sufficiently large new grid and fill it
        # A slightly larger canvas might be needed for certain offsets to prevent clipping.
        canvas_height = height + width + abs(height - 1) + 2 # heuristic for enough space
        canvas_width = height + width + abs(width - 1) + 2

        new_grid = np.full((canvas_height, canvas_width), fill_value, dtype=grid_np.dtype)

        # Offsets to center the transformed grid or align it as per specific ARC-interpretations.
        # These are based on analysis of common ARC-45-degree transformations.
        # These might need tuning depending on precise task requirements.
        # Defaulting to 0,0 and adjusting based on size
        center_row_offset = (canvas_height - new_height_raw) // 2
        center_col_offset = (canvas_width - new_width_raw) // 2


        # Iterate through each cell of the original grid
        for row_orig in range(height):
            for col_orig in range(width):
                if grid_np[row_orig, col_orig] != fill_value:
                    value = grid_np[row_orig, col_orig]

                    # Apply the specific coordinate transformation
                    # This mapping tends to create a diamond shape
                    new_row = row_orig - col_orig + (width - 1) + center_row_offset
                    new_col = row_orig + col_orig + center_col_offset

                    # Place the original value at the new calculated coordinates
                    if 0 <= new_row < canvas_height and 0 <= new_col < canvas_width:
                        new_grid[new_row, new_col] = value

        # Crop the new_grid to remove empty rows/columns on the edges
        non_fill_coords = np.argwhere(new_grid != fill_value)
        if non_fill_coords.size == 0:
            return [[fill_value]] # Return a single cell if the grid becomes empty

        min_r, min_c = non_fill_coords.min(axis=0)
        max_r, max_c = non_fill_coords.max(axis=0)

        # Add a small buffer around the cropped area to ensure no values are cut off
        # and to provide minimal visual separation. Adjust as needed.
        buffer = 0
        min_r = max(0, min_r - buffer)
        min_c = max(0, min_c - buffer)
        max_r = min(new_grid.shape[0] - 1, max_r + buffer)
        max_c = min(new_grid.shape[1] - 1, max_c + buffer)

        cropped_grid = new_grid[min_r : max_r + 1, min_c : max_c + 1]

        return cropped_grid.tolist()

    @staticmethod
    def rotate_45(grid, fill_value=0):
        """
        Applies a 45-degree diagonal transformation to the grid.
        This method acts as an alias for `transform_grid_45_degree_diagonal`
        as per the user's request to add a '45degree rotational method'.
        """
        return GridTransformer.transform_grid_45_degree_diagonal(grid, fill_value)

    def get_all_transformations(self, grid):
        """
        Applies all specified transformations to a grid and returns them in a dictionary.
        Returns:
            A dictionary with transformation names as keys and the transformed grids as values.
        """
        if not grid or not grid[0]: # Handle empty grid input
            return {}

        original_grid = grid
        inverted_grid = self.invert_colors(grid)

        transformations = {
            "Original": original_grid,
            "Rotated_90_deg": self.rotate_90(original_grid),
            "Rotated_180_deg": self.rotate_180(original_grid),
            "Rotated_270_deg": self.rotate_270(original_grid),
            "Flipped_Horizontally": self.flip_horizontal(original_grid),
            "Flipped_Vertically": self.flip_vertical(original_grid),
            "Flipped_Main_Diagonal": self.flip_main_diagonal(original_grid),
            "Flipped_Anti-Diagonal": self.flip_anti_diagonal(original_grid),
            "Rotated_45_deg": self.rotate_45(original_grid),
            "Inverted": inverted_grid,
            "Inverted_Rotated_90_deg": self.rotate_90(inverted_grid),
            "Inverted_Rotated_180_deg": self.rotate_180(inverted_grid),
            "Inverted_Rotated_270_deg": self.rotate_270(inverted_grid),
            "Inverted_Flipped_Horizontally": self.flip_horizontal(inverted_grid),
            "Inverted_Flipped_Vertically": self.flip_vertical(inverted_grid),
            "Inverted_Flipped_Main_Diagonal": self.flip_main_diagonal(inverted_grid),
            "Inverted_Flipped_Anti-Diagonal": self.flip_anti_diagonal(inverted_grid),
            "Inverted_Rotated_45_deg": self.rotate_45(inverted_grid)
        }
        return transformations

class DataSaver:
    """
    A class to save data to a JSON file.
    """
    @staticmethod
    def save_to_json(data, file_path, indent=4):
        """Saves a Python dictionary to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            # print(f"Data successfully saved to: {file_path}") # Commented for less verbose output
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")

def plot_grids(title, input_grid, output_grid):
    """Plots input and output grids with their corresponding title."""
    input_grid_np = np.array(input_grid)
    output_grid_np = np.array(output_grid)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    fig.suptitle(title, fontsize=12, fontweight='bold', color='#eeeeee', y=1.05)

    axs[0].imshow(input_grid_np, cmap=cmap, norm=norm)
    axs[0].set_title('Input', color='#dddddd')
    # Use grid_np.shape for dimension consistency, handle empty grids
    if input_grid_np.shape[1] > 0:
        axs[0].set_xticks(np.arange(-.5, input_grid_np.shape[1], 1))
    if input_grid_np.shape[0] > 0:
        axs[0].set_yticks(np.arange(-.5, input_grid_np.shape[0], 1))
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].grid(visible=True, color='#666666', linewidth=0.8)

    axs[1].imshow(output_grid_np, cmap=cmap, norm=norm)
    axs[1].set_title('Output', color='#dddddd')
    if output_grid_np.shape[1] > 0:
        axs[1].set_xticks(np.arange(-.5, output_grid_np.shape[1], 1))
    if output_grid_np.shape[0] > 0:
        axs[1].set_yticks(np.arange(-.5, output_grid_np.shape[0], 1))
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].grid(visible=True, color='#666666', linewidth=0.8)

    fig.patch.set_facecolor('#444444')
    plt.tight_layout()
    plt.show()


class DatasetTransformer:
    """
    A class to manage the processing of ARC-AGI datasets, applying transformations
    and saving the results to a specified directory structure.
    """
    def __init__(self, base_data_path):
        """
        Initializes the DatasetTransformer with the base path for the data.

        Args:
            base_data_path (str): The root directory where the dataset is located.
        """
        self.base_data_path = base_data_path
        self.transformer = GridTransformer()
        self.data_saver = DataSaver()

    def process_and_save_all_transformations(self, file_path, base_output_dir):
        """
        Loads and processes a single task or a list of tasks from a JSON file,
        applies all geometric transformations, and saves the results.
        Handles both dictionary and list structures loaded from JSON.
        """
        try:
            task_data_all = load_json(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return

        # Extract the task ID from the filename (e.g., 'a740d043.json' -> 'a740d043')
        task_id = Path(file_path).stem

        # Check if the loaded data is a list of tasks
        if isinstance(task_data_all, list):
            tasks_to_process = task_data_all
        elif isinstance(task_data_all, dict):
            # Check for the old format ('train' and 'test' keys)
            if 'train' in task_data_all and task_data_all['train']:
                # Handle the old format (nested train/test keys), treat the first train example as the task
                if task_data_all['train']:
                    tasks_to_process = [task_data_all['train'][0]]
                else:
                    print(f"Skipping Task ID: {task_id} as it has no 'train' examples in {file_path}.")
                    return
            else:
                # Handle the new, simpler format (direct input/output keys)
                # Treat the dictionary itself as a single task
                tasks_to_process = [task_data_all]
        else:
            print(f"Skipping Task ID: {task_id} due to unsupported JSON structure in {file_path}.")
            return

        for task_data in tasks_to_process:
            input_grid = task_data.get('input', [])
            output_grid = task_data.get('output', [])

            if not input_grid:
                # If there's no input grid, it's an invalid task for this purpose
                print(f"Skipping Task ID: {task_id} within {file_path} as it has no valid 'input' grid.")
                continue # Move to the next task in the list if applicable

            # Ensure input_grid is not empty for transformation
            if not input_grid or not input_grid[0]:
                print(f"Skipping Task ID: {task_id} within {file_path} due to empty input grid.")
                continue

            # Get transformation methods dynamically. Use a small grid as placeholder for keys.
            transformation_methods = list(self.transformer.get_all_transformations([[0]]).keys())

            transformed_input_data = self.transformer.get_all_transformations(input_grid)
            transformed_output_data = self.transformer.get_all_transformations(output_grid)


            for method in transformation_methods:
                output_dir = Path(base_output_dir) / method
                output_dir.mkdir(parents=True, exist_ok=True)

                transformed_data = {
                    'task_id': task_id,
                    'input': transformed_input_data.get(method, []),
                    'output': transformed_output_data.get(method, [])
                }

                # Modify the output filename to include an index if task_data_all was a list
                output_file_name = f'{task_id}_transformed_{method}.json'
                if isinstance(task_data_all, list) and len(tasks_to_process) > 1:
                    task_index = tasks_to_process.index(task_data)
                    output_file_name = f'{task_id}_{task_index}_transformed_{method}.json'


                output_file_path = output_dir / output_file_name
                self.data_saver.save_to_json(transformed_data, output_file_path)


    def run(self):
        """
        Executes the transformation and saving process for the dataset.
        Modified to look for all files directly in the specified data path.
        """
        print("Starting data transformation for the dataset.")

        # Process all challenge files in the specified directory
        path_pattern = f"{self.base_data_path}/*.json"
        challenge_files = glob.glob(path_pattern)
        output_dir_path = Path(self.base_data_path) / 'GridTransitionDataset'

        if not challenge_files:
            print(f"No files found at '{path_pattern}'. Please check your path and Google Drive mount.")
        else:
            print(f"Processing {len(challenge_files)} files...")
            for file_path in tqdm(challenge_files): # Wrap challenge_files with tqdm
                self.process_and_save_all_transformations(file_path, output_dir_path)
            print(f"Finished processing files. Transformed data saved to {output_dir_path}")

        print("\nAll transformations completed.")


if __name__ == '__main__':
    # Adjust this path to the correct location of your JSON files
    base_data_path = '/content/drive/MyDrive/Google_AI_Studio/ARCAGI2025/data/tasks'
    processor = DatasetTransformer(base_data_path)
    processor.run()

"""# Task
Optimize the provided Python code to reduce the execution time of the data transformation process from approximately 346 seconds per iteration to 15 seconds per iteration. Identify and address performance bottlenecks, potentially using profiling tools and parallel processing.

## Detailed profiling

### Subtask:
Use a more detailed profiler (like `line_profiler` if installed, or analyze `cProfile` output carefully) to pinpoint the exact lines of code within the transformation functions that are taking the most time.

**Reasoning**:
The first step is to install `line_profiler` which is required to profile the code.
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install line_profiler

"""**Reasoning**:
Import the `profile` decorator from `line_profiler` and modify the `DatasetTransformer` class to apply the `@profile` decorator to the `process_and_save_all_transformations` method for profiling.


"""

from line_profiler import profile

class DataSaver:
    """
    A class to save data to a JSON file.
    """
    @staticmethod
    def save_to_json(data, file_path, indent=4):
        """Saves a Python dictionary to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            # print(f"Data successfully saved to: {file_path}") # Commented for less verbose output
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")

def plot_grids(title, input_grid, output_grid):
    """Plots input and output grids with their corresponding title."""
    input_grid_np = np.array(input_grid)
    output_grid_np = np.array(output_grid)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    fig.suptitle(title, fontsize=12, fontweight='bold', color='#eeeeee', y=1.05)

    axs[0].imshow(input_grid_np, cmap=cmap, norm=norm)
    axs[0].set_title('Input', color='#dddddd')
    # Use grid_np.shape for dimension consistency, handle empty grids
    if input_grid_np.shape[1] > 0:
        axs[0].set_xticks(np.arange(-.5, input_grid_np.shape[1], 1))
    if input_grid_np.shape[0] > 0:
        axs[0].set_yticks(np.arange(-.5, input_grid_np.shape[0], 1))
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].grid(visible=True, color='#666666', linewidth=0.8)

    axs[1].imshow(output_grid_np, cmap=cmap, norm=norm)
    axs[1].set_title('Output', color='#dddddd')
    if output_grid_np.shape[1] > 0:
        axs[1].set_xticks(np.arange(-.5, output_grid_np.shape[1], 1))
    if output_grid_np.shape[0] > 0:
        axs[1].set_yticks(np.arange(-.5, output_grid_np.shape[0], 1))
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].grid(visible=True, color='#666666', linewidth=0.8)

    fig.patch.set_facecolor('#444444')
    plt.tight_layout()
    plt.show()


class DatasetTransformer:
    """
    A class to manage the processing of ARC-AGI datasets, applying transformations
    and saving the results to a specified directory structure.
    """
    def __init__(self, base_data_path):
        """
        Initializes the DatasetTransformer with the base path for the data.

        Args:
            base_data_path (str): The root directory where the dataset is located.
        """
        self.base_data_path = base_data_path
        self.transformer = GridTransformer()
        self.data_saver = DataSaver()

    @profile
    def process_and_save_all_transformations(self, file_path, base_output_dir):
        """
        Loads and processes a single task or a list of tasks from a JSON file,
        applies all geometric transformations, and saves the results.
        Handles both dictionary and list structures loaded from JSON.
        """
        try:
            task_data_all = load_json(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return

        # Extract the task ID from the filename (e.g., 'a740d043.json' -> 'a740d043')
        task_id = Path(file_path).stem

        # Check if the loaded data is a list of tasks
        if isinstance(task_data_all, list):
            tasks_to_process = task_data_all
        elif isinstance(task_data_all, dict):
            # Check for the old format ('train' and 'test' keys)
            if 'train' in task_data_all and task_data_all['train']:
                # Handle the old format (nested train/test keys), treat the first train example as the task
                if task_data_all['train']:
                    tasks_to_process = [task_data_all['train'][0]]
                else:
                    print(f"Skipping Task ID: {task_id} as it has no 'train' examples in {file_path}.")
                    return
            else:
                # Handle the new, simpler format (direct input/output keys)
                # Treat the dictionary itself as a single task
                tasks_to_process = [task_data_all]
        else:
            print(f"Skipping Task ID: {task_id} due to unsupported JSON structure in {file_path}.")
            return

        for task_data in tasks_to_process:
            input_grid = task_data.get('input', [])
            output_grid = task_data.get('output', [])

            if not input_grid:
                # If there's no input grid, it's an invalid task for this purpose
                print(f"Skipping Task ID: {task_id} within {file_path} as it has no valid 'input' grid.")
                continue # Move to the next task in the list if applicable

            # Ensure input_grid is not empty for transformation
            if not input_grid or not input_grid[0]:
                print(f"Skipping Task ID: {task_id} within {file_path} due to empty input grid.")
                continue

            # Get transformation methods dynamically. Use a small grid as placeholder for keys.
            transformation_methods = list(self.transformer.get_all_transformations([[0]]).keys())

            transformed_input_data = self.transformer.get_all_transformations(input_grid)
            transformed_output_data = self.transformer.get_all_transformations(output_grid)


            for method in transformation_methods:
                output_dir = Path(base_output_dir) / method
                output_dir.mkdir(parents=True, exist_ok=True)

                transformed_data = {
                    'task_id': task_id,
                    'input': transformed_input_data.get(method, []),
                    'output': transformed_output_data.get(method, [])
                }

                # Modify the output filename to include an index if task_data_all was a list
                output_file_name = f'{task_id}_transformed_{method}.json'
                if isinstance(task_data_all, list) and len(tasks_to_process) > 1:
                    task_index = tasks_to_process.index(task_data)
                    output_file_name = f'{task_id}_{task_index}_transformed_{method}.json'


                output_file_path = output_dir / output_file_name
                self.data_saver.save_to_json(transformed_data, output_file_path)


    def run(self):
        """
        Executes the transformation and saving process for the dataset.
        Modified to look for all files directly in the specified data path.
        """
        print("Starting data transformation for the dataset.")

        # Process all challenge files in the specified directory
        path_pattern = f"{self.base_data_path}/*.json"
        challenge_files = glob.glob(path_pattern)
        output_dir_path = Path(self.base_data_path) / 'GridTransitionDataset'

        if not challenge_files:
            print(f"No files found at '{path_pattern}'. Please check your path and Google Drive mount.")
        else:
            print(f"Processing {len(challenge_files)} files...")
            for file_path in tqdm(challenge_files): # Wrap challenge_files with tqdm
                self.process_and_save_all_transformations(file_path, output_dir_path)
            print(f"Finished processing files. Transformed data saved to {output_dir_path}")

        print("\nAll transformations completed.")

"""**Reasoning**:
Now that `line_profiler` is installed and the `@profile` decorator is added, run the `DatasetTransformer.run()` method to generate the profiling data.


"""

if __name__ == '__main__':
    # Adjust this path to the correct location of your JSON files
    base_data_path = '/Users/seungwonlee/ARCprize2025_HIN/re-ARC-dataset/tasks'
    processor = DatasetTransformer(base_data_path)
    # Only process a subset of files for faster profiling
    processor.run()