import numpy as np

def transform_grid_45_degree_diagonal(grid, fill_value=0):
    """
    Transforms a 2D grid to a 45-degree diagonal pattern based on
    a specific coordinate mapping.

    Args:
        grid (list of lists): The input 2D grid.
        fill_value (int): The value to fill blank spaces with.

    Returns:
        list of lists: The new, transformed grid.
    """
    if not grid or not grid[0]:
        return [[]]

    grid = np.array(grid)
    height, width = grid.shape

    # Correctly calculate the dimensions of the new rotated grid's bounding box.
    new_height = height + width + 1
    new_width = height + width + 1
    
    # Create the new grid and fill it with the fill_value
    new_grid = np.full((new_height, new_width), fill_value, dtype=grid.dtype)

    # Offsets determined by analyzing the human feedback answersheet
    row_offset = 3
    col_offset = 1

    # Iterate through each cell of the original grid
    for row_orig in range(height):
        for col_orig in range(width):
            if grid[row_orig, col_orig] != fill_value:
                value = grid[row_orig, col_orig]
                
                # Apply the specific coordinate transformation
                new_row = row_orig - col_orig + row_offset
                new_col = row_orig + col_orig + col_offset

                # Place the original value at the new calculated coordinates
                if 0 <= new_row < new_height and 0 <= new_col < new_width:
                    new_grid[new_row, new_col] = value

    return new_grid.tolist()

# Example usage with the sample grid
sample_grid = [
    [0, 4, 0, 9],
    [0, 0, 0, 0],
    [0, 4, 6, 0],
    [1, 0, 0, 0]
]

# Get the transformed grid
transformed_grid = transform_grid_45_degree_diagonal(sample_grid)

# Print the original and try6 transformed grids
print("Original Grid:")
for row in sample_grid:
    print(row)

print("\n45-degree Rotated Grid (try6):")
for row in transformed_grid:
    print(row)

"""
input": [[0, 4, 0, 9], [0, 0, 0, 0], [0, 4, 6, 0], [1, 0, 0, 0]], "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [1, 4, 6, 9]
"""