import sys
import subprocess
from discrete_solver import ZBDDManager, SetCoverSolver, KnapsackSolver, NQueensSolver, to_ascii, to_svg

def run_tests():
    print("=" * 60)
    print("1. RUNNING AUTOMATED UNIT TESTS VIA PYTEST")
    print("=" * 60)
    try:
        res = subprocess.run(["python", "-m", "pytest", "tests/test_zbdd.py", "-v"], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("✨ All unit tests passed successfully!\n")
            return True
        else:
            print("❌ Some unit tests failed:")
            print(res.stderr)
            return False
    except Exception as e:
        print(f"⚠️ Could not execute pytest: {e}\n")
        # Run manually as a fallback
        return run_tests_manually()

def run_tests_manually():
    print("Falling back to manual test execution...")
    # Import and run test functions directly
    sys.path.append('.')
    from tests.test_zbdd import (
        test_zbdd_basics, test_zero_suppression, test_set_operations,
        test_algebraic_identities, test_division, test_change_and_subsets,
        test_symmetric_and_size_count, test_min_weight_and_knapsack,
        test_set_cover_solver, test_knapsack_solver, test_nqueens_solver
    )
    tests = [
        test_zbdd_basics, test_zero_suppression, test_set_operations,
        test_algebraic_identities, test_division, test_change_and_subsets,
        test_symmetric_and_size_count, test_min_weight_and_knapsack,
        test_set_cover_solver, test_knapsack_solver, test_nqueens_solver
    ]
    all_pass = True
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__} passed.")
        except Exception as ex:
            print(f"  ❌ {t.__name__} failed: {ex}")
            all_pass = False
    return all_pass

def demo_set_cover():
    print("=" * 60)
    print("2. DEMONSTRATION: SET COVER SOLVER")
    print("=" * 60)
    # Universe of skills to cover
    universe = {"Logic", "Search", "V-AE", "RL", "Data", "Optimization"}
    subsets = {
        "ML_Module": {"V-AE", "RL", "Data"},
        "Discrete_Module": {"Logic", "Search", "Optimization"},
        "Theory_Module": {"Logic", "V-AE", "Optimization"},
        "Applied_Module": {"Search", "RL", "Data"},
        "Expert_Module": {"RL", "Optimization"}
    }
    weights = {
        "ML_Module": 15.0,
        "Discrete_Module": 12.0,
        "Theory_Module": 8.0,
        "Applied_Module": 10.0,
        "Expert_Module": 5.0
    }
    
    print("Input Universe:", list(universe))
    print("Available Modules & Coverage & Cost:")
    for k, v in subsets.items():
        print(f"  - {k:15}: Covers {list(v)} with cost {weights[k]}")
        
    solver = SetCoverSolver(universe, subsets, weights)
    res = solver.solve()
    
    if res:
        print("\nOptimal Solution Found:")
        print(f"  Selected Modules: {res['cover']}")
        print(f"  Total Cost:       {res['cost']}")
        print(f"  Total valid covers represented in ZBDD: {res['total_valid_covers']}")
        print(f"  ZBDD node count:  {res['zbdd_nodes']}")
    else:
        print("❌ No valid set cover possible.")
    print()

def demo_knapsack():
    print("=" * 60)
    print("3. DEMONSTRATION: ATTRIBUTED ZBDD KNAPSACK SOLVER")
    print("=" * 60)
    items = ["item_A", "item_B", "item_C", "item_D", "item_E"]
    weights = {"item_A": 4, "item_B": 2, "item_C": 3, "item_D": 5, "item_E": 1}
    values = {"item_A": 10, "item_B": 6, "item_C": 8, "item_D": 12, "item_E": 3}
    capacity = 7
    
    print(f"Capacity limit: {capacity}")
    print("Items list:")
    for item in items:
        print(f"  - {item:8}: Weight {weights[item]}, Value {values[item]}")
        
    solver = KnapsackSolver(items, weights, values, capacity)
    res = solver.solve()
    
    if res:
        print("\nOptimal Knapsack Selection:")
        print(f"  Selected Items:   {res['selection']}")
        print(f"  Total Value:      {res['value']}")
        print(f"  Total Weight:     {res['weight']}")
        print(f"  ZBDD state-space size (nodes): {res['zbdd_nodes']}")
    else:
        print("❌ No selection fits capacity constraints.")
    print()

def demo_nqueens():
    print("=" * 60)
    print("4. DEMONSTRATION: N-QUEENS SOLVER (N=4)")
    print("=" * 60)
    solver = NQueensSolver(4)
    res = solver.solve()
    
    print(f"For a 4x4 chessboard:")
    print(f"  Number of unique solutions: {res['count']}")
    print(f"  Number of ZBDD nodes:       {res['zbdd_nodes']}")
    print("\nVisualizing the 2 Solutions:")
    for idx, grid in enumerate(res['solutions']):
        print(f"\nSolution {idx + 1}:")
        for row in grid:
            print(" ".join("♛" if cell == 1 else "·" for cell in row))
            
    # Print the ASCII representation of the 4-Queens solution ZBDD
    print("\nZBDD Structural Representation (ASCII tree):")
    # To get the final ZBDD node ID, we compile the constraints again
    manager = solver.manager
    print(to_ascii(manager, len(manager.nodes) - 1))
    
    # Save the ZBDD visual graph as an SVG artifact
    svg_filename = "zbdd_4queens_solutions.svg"
    to_svg(manager, len(manager.nodes) - 1, filename=svg_filename)
    print(f"\n🎨 SVG visualization saved successfully as '{svg_filename}'!")
    print()

if __name__ == "__main__":
    test_success = run_tests()
    demo_set_cover()
    demo_knapsack()
    demo_nqueens()
    if not test_success:
        sys.exit(1)
