# ZBDD-Based Discrete Logic Solver Engine Core

A high-performance, proprietary Python implementation of a **Zero-suppressed Binary Decision Diagram (ZBDD / ZDD)** engine core, tailored for discrete structure optimization, constraint satisfaction, and combinatorial search.

---

## 🧠 ZBDD Mathematical Foundation

ZBDDs, first introduced by Shin-ichi Minato in 1993, are a variant of Binary Decision Diagrams (BDDs) designed to represent **families of sets** (combination sets) extremely compactly. 

Unlike standard BDDs, which apply the Shannon reduction rule to eliminate redundant nodes where `low == high`, ZBDDs apply the **Zero-suppression rule**:
* **Zero-suppression rule**: A node is eliminated if its `high` branch (which represents the choice of including the variable in the combination) points to the terminal `0` (the empty set family $\emptyset$).

This simple change makes ZBDDs exponentially more compact for representing sparse sets or combination structures compared to standard BDDs.

### Set Family Interpretation
- Terminal node **`0`** represents the empty family of sets: $\emptyset$.
- Terminal node **`1`** represents the family containing only the empty set: $\{\emptyset\}$.
- An internal node **`N = (x, L, H)`** represents the family of sets:
  $$S = L \cup \{ s \cup \{x\} \mid s \in H \}$$

---

## 🛠️ Package Structure

The package is organized as follows:
```
discrete_solver/
├── __init__.py          # Exports the public API
├── zbdd.py              # Core ZBDD node management, unique table, caches, and algorithms
├── solver.py            # High-level optimization solvers (Set Cover, Knapsack, N-Queens)
└── visualizer.py        # Exports DOT, ASCII, and standalone SVG layout files
```

---

## 🚀 Key Features

1. **Canonical Node Management**: A central unique table ensures no duplicate nodes are created, giving a canonical representation of set families.
2. **Computed Caches**: Operates on sub-graphs using memoized computation caches to guarantee polynomial-time complexity for set operations.
3. **Advanced Set Algebra**:
   - `union(P, Q)`: $P \cup Q$
   - `intersection(P, Q)`: $P \cap Q$
   - `difference(P, Q)`: $P \setminus Q$
   - `product(P, Q)`: Pairwise set union $P \times Q$ (Cartesian set product)
   - `quotient(P, Q)` and `remainder(P, Q)`: Algebraic division of set families.
   - `change(P, x)`: Toggles variable $x$'s presence in the family.
   - `symmetric_k_n(vars, k)`: Compiles the exact combinatorics of choosing $k$ variables.
4. **Attributed Graph Optimization**:
   - `find_min_weight_set(P, weights)`: Solves the shortest path problem in $O(|P|)$ time over the ZBDD graph.
   - `find_max_weight_set(P, weights, capacity, values)`: Solves the Knapsack problem over arbitrary set constraints.
   - `count_sets_by_size(P)`: Computes the size distribution of sets.
5. **Robust Exporters**:
   - Beautiful ASCII console prints of the decision diagram.
   - Standalone SVG generator placing nodes in horizontal layers corresponding to variable ordering.

---

## 📖 Usage Examples

### 1. Basic Set Algebra
```python
from discrete_solver import ZBDDManager

manager = ZBDDManager()

# Create sets {{x}} and {{y}}
x = manager.element('x')
y = manager.element('y')

# Union: {{x}, {y}}
U = manager.union(x, y)

# Product: {{x, y}}
P = manager.product(x, y)

print(manager.get_sets(U)) # [{'x'}, {'y'}]
print(manager.get_sets(P)) # [{'x', 'y'}]
```

### 2. High-Level Solvers

#### Minimum Cost Set Cover
```python
from discrete_solver import SetCoverSolver

universe = {1, 2, 3, 4, 5}
subsets = {
    'S1': {1, 2, 3},
    'S2': {3, 4},
    'S3': {4, 5},
    'S4': {1, 5}
}
weights = {
    'S1': 2.0,
    'S2': 1.0,
    'S3': 1.0,
    'S4': 3.0
}

solver = SetCoverSolver(universe, subsets, weights)
result = solver.solve()
print("Selected subsets:", result['cover']) # {'S1', 'S3'}
print("Total cost:", result['cost'])         # 3.0
```

#### Constraint Knapsack
```python
from discrete_solver import KnapsackSolver

items = ['A', 'B', 'C', 'D']
weights = {'A': 2, 'B': 3, 'C': 4, 'D': 5}
values = {'A': 3, 'B': 4, 'C': 5, 'D': 8}
capacity = 8

solver = KnapsackSolver(items, weights, values, capacity)
result = solver.solve()
print("Chosen items:", result['selection']) # {'B', 'D'}
print("Total value:", result['value'])      # 12
```

#### N-Queens constraint Solver
```python
from discrete_solver import NQueensSolver

solver = NQueensSolver(8)
result = solver.solve()
print("Number of valid 8-Queens boards:", result['count']) # 92
```

---

## 🎨 Visualization Exporters

Export a ZBDD as a layered SVG node diagram:
```python
from discrete_solver import to_svg

# Given a manager and a ZBDD node ID P:
to_svg(manager, P, filename="diagram.svg")
```
Or export a Graphviz DOT representation:
```python
from discrete_solver import to_dot

dot_str = to_dot(manager, P)
```
