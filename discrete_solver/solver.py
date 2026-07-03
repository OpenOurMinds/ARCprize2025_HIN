from discrete_solver.zbdd import ZBDDManager

def make_power_set(manager, vars_list):
    """
    Creates the power set ZBDD for a given list of variables.
    Constructs a canonical ZBDD representing all 2^N subsets in O(N) nodes.
    """
    # Sort variables according to ordering to ensure canonical construction
    sorted_vars = sorted(vars_list, key=lambda v: manager._register_var(v))
    memo = {}
    
    def build(idx):
        if idx == len(sorted_vars):
            return 1
        if idx in memo:
            return memo[idx]
            
        rest = build(idx + 1)
        res = manager.get_node(sorted_vars[idx], rest, rest)
        memo[idx] = res
        return res
        
    return build(0)

class SetCoverSolver:
    """
    Solves the exact and minimum-weight Set Cover problem using ZBDDs.
    Encodes constraints in ZBDD structure and queries for the minimum weight set.
    """
    def __init__(self, universe, subsets, weights=None):
        self.universe = list(universe)
        self.subsets = subsets  # Dict of subset_name -> list/set of elements
        self.weights = weights if weights is not None else {name: 1.0 for name in subsets}
        self.manager = ZBDDManager()
        
    def solve(self):
        subset_names = list(self.subsets.keys())
        for name in subset_names:
            self.manager._register_var(name)
            
        # Build power set of all subsets (representing all choices of sets)
        all_power_set = make_power_set(self.manager, subset_names)
        
        valid_covers = all_power_set
        
        # Intersect constraint ZBDDs for each element of the universe
        for element in self.universe:
            # Subsets that cover the element
            covering = [name for name, items in self.subsets.items() if element in items]
            if not covering:
                # Impossible to cover this element
                return None
                
            # Subsets that DO NOT cover the element
            non_covering = [name for name, items in self.subsets.items() if element not in items]
            
            # Subsets of non_covering are selections that FAIL to cover this element
            invalid_power_set = make_power_set(self.manager, non_covering)
            
            # Valid covers for this element are (All - Invalid)
            covering_constraint = self.manager.difference(all_power_set, invalid_power_set)
            
            # Intersect with overall solution space
            valid_covers = self.manager.intersection(valid_covers, covering_constraint)
            
        # Extract the minimum cost cover
        min_cost, best_cover = self.manager.find_min_weight_set(valid_covers, self.weights)
        if best_cover is None:
            return None
            
        return {
            'cost': min_cost,
            'cover': best_cover,
            'total_valid_covers': self.manager.count_sets(valid_covers),
            'zbdd_nodes': self.manager.count_nodes(valid_covers)
        }

class KnapsackSolver:
    """
    Solves the 0/1 Knapsack problem using ZBDD memoized search.
    Builds the state-space of choices and runs a pseudo-polynomial time search on the DAG.
    """
    def __init__(self, items, weights, values, capacity):
        self.items = list(items)
        self.weights = weights  # Dict of item -> weight
        self.values = values    # Dict of item -> value
        self.capacity = capacity
        self.manager = ZBDDManager()
        
    def solve(self):
        for item in self.items:
            self.manager._register_var(item)
            
        # Build power set representing all possible item subsets
        all_subsets = make_power_set(self.manager, self.items)
        
        # Query ZBDD for best configuration under capacity constraint
        max_value, best_selection = self.manager.find_max_weight_set(
            all_subsets, self.weights, self.capacity, self.values
        )
        
        if best_selection is None:
            return None
            
        total_weight = sum(self.weights[item] for item in best_selection)
        
        return {
            'value': max_value,
            'selection': best_selection,
            'weight': total_weight,
            'zbdd_nodes': self.manager.count_nodes(all_subsets)
        }

class NQueensSolver:
    """
    Solves the N-Queens problem using ZBDDs.
    Generates row combinations and filters out column/diagonal collisions.
    """
    def __init__(self, n):
        self.n = n
        self.manager = ZBDDManager()
        
    def solve(self):
        board_vars = []
        rows = [[] for _ in range(self.n)]
        cols = [[] for _ in range(self.n)]
        diag1 = {}  # r - c constant
        diag2 = {}  # r + c constant
        
        for r in range(self.n):
            for c in range(self.n):
                var = f"{r}_{c}"
                board_vars.append(var)
                rows[r].append(var)
                cols[c].append(var)
                
                d1 = r - c
                if d1 not in diag1:
                    diag1[d1] = []
                diag1[d1].append(var)
                
                d2 = r + c
                if d2 not in diag2:
                    diag2[d2] = []
                diag2[d2].append(var)
                
        # Register variables in board order
        for var in board_vars:
            self.manager._register_var(var)
            
        # 1. Start with Row constraint: exactly 1 queen in each row.
        # This is the Cartesian product of (exactly 1 queen in row r) across all rows.
        solutions = 1  # Base family {∅}
        for r in range(self.n):
            row_choices = self.manager.symmetric_k_n(rows[r], 1)
            solutions = self.manager.product(solutions, row_choices)
            
        # 2. Filter Column constraint: at most 1 queen in each column
        for c in range(self.n):
            solutions = self.manager.filter_at_most_k(solutions, cols[c], 1)
            
        # 3. Filter diagonal constraints: at most 1 queen in each diagonal
        for d1, vars_list in diag1.items():
            if len(vars_list) > 1:
                solutions = self.manager.filter_at_most_k(solutions, vars_list, 1)
                
        for d2, vars_list in diag2.items():
            if len(vars_list) > 1:
                solutions = self.manager.filter_at_most_k(solutions, vars_list, 1)
                
        # 4. Gather results
        num_solutions = self.manager.count_sets(solutions)
        all_raw_solutions = self.manager.get_sets(solutions)
        
        board_solutions = []
        for raw_sol in all_raw_solutions:
            grid = [[0] * self.n for _ in range(self.n)]
            for queen in raw_sol:
                r, c = map(int, queen.split('_'))
                grid[r][c] = 1
            board_solutions.append(grid)
            
        return {
            'solutions': board_solutions,
            'count': num_solutions,
            'zbdd_nodes': self.manager.count_nodes(solutions)
        }
