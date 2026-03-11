# arc_grid_transformer/test_arctransformer.py

import sys
import os

# Add the directory containing the compiled Rust library to Python's path
# This assumes you are running this script from the 'arc_grid_transformer' directory
# and the Rust library compiled into 'arctransformer/target/release'
# Adjust this path if your setup is different.
rust_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'arctransformer', 'target', 'release'))
sys.path.insert(0, rust_lib_path)

try:
    # Now you can import your Rust-backed module and classes
    import arctransformer
    from arctransformer import GridTransformerRust
except ImportError as e:
    print(f"Error importing arctransformer: {e}")
    print(f"Make sure the compiled library (e.g., arctransformer.cpython-3x-y.so or .pyd) is in: {rust_lib_path}")
    sys.exit(1)

def print_grid(name, grid):
    print(f"\n--- {name} ({len(grid)}x{len(grid[0]) if grid else 0}) ---")
    if not grid:
        print("  (Empty Grid)")
        return
    for row in grid:
        print("  ", row)

# Define an example grid
example_grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Another example for 45-degree rotation
diag_grid = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]

print("--- Testing Individual GridTransformerRust Methods ---")

# Use the static methods directly
try:
    print_grid("Original Grid", example_grid)

    rotated_90 = GridTransformerRust.rotate_90(example_grid)
    print_grid("Rotated 90 degrees", rotated_90)

    rotated_180 = GridTransformerRust.rotate_180(example_grid)
    print_grid("Rotated 180 degrees", rotated_180)

    rotated_270 = GridTransformerRust.rotate_270(example_grid)
    print_grid("Rotated 270 degrees", rotated_270)

    flipped_h = GridTransformerRust.flip_horizontal(example_grid)
    print_grid("Flipped Horizontally", flipped_h)

    flipped_v = GridTransformerRust.flip_vertical(example_grid)
    print_grid("Flipped Vertically", flipped_v)

    flipped_main_diag = GridTransformerRust.flip_main_diagonal(example_grid)
    print_grid("Flipped Main Diagonal", flipped_main_diag)

    flipped_anti_diag = GridTransformerRust.flip_anti_diagonal(example_grid)
    print_grid("Flipped Anti-Diagonal", flipped_anti_diag)

    inverted_grid = GridTransformerRust.invert_colors(example_grid)
    print_grid("Inverted Colors", inverted_grid)

    print_grid("Original Diagonal Grid", diag_grid)
    rotated_45 = GridTransformerRust.rotate_45(diag_grid, 0) # fill_value=0
    print_grid("Rotated 45 degrees (diag_grid)", rotated_45)

    # Test with empty grid
    empty_grid_result = GridTransformerRust.rotate_90([])
    print_grid("Empty Grid (rotated 90)", empty_grid_result)
    empty_45_deg_result = GridTransformerRust.rotate_45([], 0)
    print_grid("Empty Grid (rotated 45)", empty_45_deg_result)

except Exception as e:
    print(f"\nError during individual method testing: {e}")

print("\n--- Testing get_all_transformations_rust function ---")

# Test the combined function
try:
    all_transforms = arctransformer.get_all_transformations_rust(example_grid)
    print(f"Total transformations generated: {len(all_transforms)}")
    for name, grid in all_transforms.items():
        if "Original" in name: # print original first for comparison
             print_grid(name, grid)
        # print_grid(name, grid) # Uncomment to print all grids

    # Example of printing a specific transformation
    print_grid("Specific: Inverted_Rotated_90_deg", all_transforms["Inverted_Rotated_90_deg"])

    # Test with an empty grid
    empty_transforms = arctransformer.get_all_transformations_rust([])
    print(f"\nTotal transformations generated for empty grid: {len(empty_transforms)}")
    print_grid("Specific Empty: Original", empty_transforms["Original"])


except Exception as e:
    print(f"\nError during get_all_transformations_rust testing: {e}")

print("\n--- All tests complete ---")