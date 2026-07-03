import functools
from collections import deque

class ZBDDManager:
    """
    High-performance ZBDD Manager for representing and optimizing discrete structures.
    Maintains a unique table and computed operation caches for canonical representation 
    and efficient operations.
    """
    def __init__(self):
        # Unique table: (var_index, low_id, high_id) -> node_id
        self.unique_table = {}
        
        # Node storage: node_id -> (var_name, low_id, high_id)
        # ID 0: Empty set family (empty set, represented as terminal 0: ∅)
        # ID 1: Family containing only the empty set (base set, represented as terminal 1: {∅})
        # For terminal nodes, variable is set to float('inf') and children to None.
        self.nodes = {
            0: (float('inf'), None, None),
            1: (float('inf'), None, None)
        }
        
        # Variable ordering maps
        self.var_to_index = {}
        self.index_to_var = []
        
        # Operation cache for recursive calls to optimize performance:
        # (op_name, arg1, arg2, ...) -> result_node_id
        self.op_cache = {}
        
    def clear_cache(self):
        """Clears the computed operation cache to free memory."""
        self.op_cache.clear()
        
    def _register_var(self, var):
        """Registers a variable dynamically if not already declared."""
        if var not in self.var_to_index:
            idx = len(self.index_to_var)
            self.var_to_index[var] = idx
            self.index_to_var.append(var)
        return self.var_to_index[var]
        
    def var_index(self, node_id):
        """Returns the variable ordering index for a node ID."""
        var = self.nodes[node_id][0]
        if var == float('inf'):
            return float('inf')
        return self.var_to_index[var]

    def get_node(self, var, low, high):
        """
        Creates/retrieves a ZBDD node for (var, low, high) ensuring canonical reduction.
        Zero-suppression: If high is 0, return low.
        """
        if high == 0:
            return low
            
        # Register variable to ensure ordering
        self._register_var(var)
        
        key = (var, low, high)
        if key in self.unique_table:
            return self.unique_table[key]
            
        # Allocate new node
        node_id = len(self.nodes)
        self.nodes[node_id] = key
        self.unique_table[key] = node_id
        return node_id

    # Core set constructors
    def empty(self):
        """Returns the empty family of sets: ∅ (ZBDD Terminal 0)."""
        return 0
        
    def base(self):
        """Returns the base family containing only the empty set: {∅} (ZBDD Terminal 1)."""
        return 1
        
    def element(self, var):
        """Returns the family containing only the singleton set: {{var}}."""
        return self.get_node(var, 0, 1)

    # Set operations
    def union(self, P, Q):
        """Returns the union of set families P and Q: P ∪ Q."""
        if P == 0:
            return Q
        if Q == 0:
            return P
        if P == Q:
            return P
            
        # Commutative ordering to maximize cache hits
        if P > Q:
            P, Q = Q, P
            
        key = ('union', P, Q)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        q_var, q_low, q_high = self.nodes[Q]
        
        p_idx = self.var_index(P)
        q_idx = self.var_index(Q)
        
        if p_idx < q_idx:
            # P is above Q in the variable ordering
            res = self.get_node(p_var, self.union(p_low, Q), p_high)
        elif p_idx > q_idx:
            # Q is above P
            res = self.get_node(q_var, self.union(P, q_low), q_high)
        else:
            # Variables are identical
            res = self.get_node(p_var, self.union(p_low, q_low), self.union(p_high, q_high))
            
        self.op_cache[key] = res
        return res

    def intersection(self, P, Q):
        """Returns the intersection of set families P and Q: P ∩ Q."""
        if P == 0 or Q == 0:
            return 0
        if P == Q:
            return P
            
        # Commutative ordering
        if P > Q:
            P, Q = Q, P
            
        key = ('intersection', P, Q)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        q_var, q_low, q_high = self.nodes[Q]
        
        p_idx = self.var_index(P)
        q_idx = self.var_index(Q)
        
        if p_idx < q_idx:
            # P is above Q, so no set in Q contains p_var
            res = self.intersection(p_low, Q)
        elif p_idx > q_idx:
            # Q is above P
            res = self.intersection(P, q_low)
        else:
            res = self.get_node(p_var, self.intersection(p_low, q_low), self.intersection(p_high, q_high))
            
        self.op_cache[key] = res
        return res

    def difference(self, P, Q):
        """Returns the difference of set families P and Q: P \\ Q."""
        if P == 0:
            return 0
        if Q == 0:
            return P
        if P == Q:
            return 0
            
        key = ('difference', P, Q)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        q_var, q_low, q_high = self.nodes[Q]
        
        p_idx = self.var_index(P)
        q_idx = self.var_index(Q)
        
        if p_idx < q_idx:
            # P is above Q
            res = self.get_node(p_var, self.difference(p_low, Q), p_high)
        elif p_idx > q_idx:
            # Q is above P
            res = self.difference(P, q_low)
        else:
            res = self.get_node(p_var, self.difference(p_low, q_low), self.difference(p_high, q_high))
            
        self.op_cache[key] = res
        return res

    def product(self, P, Q):
        """
        Returns the pairwise union (Cartesian product) of set families P and Q:
        P × Q = { s ∪ t | s ∈ P, t ∈ Q }.
        """
        if P == 0 or Q == 0:
            return 0
        if P == 1:
            return Q
        if Q == 1:
            return P
            
        if P > Q:
            P, Q = Q, P
            
        key = ('product', P, Q)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        q_var, q_low, q_high = self.nodes[Q]
        
        p_idx = self.var_index(P)
        q_idx = self.var_index(Q)
        
        if p_idx < q_idx:
            res = self.get_node(p_var, self.product(p_low, Q), self.product(p_high, Q))
        elif p_idx > q_idx:
            res = self.get_node(q_var, self.product(P, q_low), self.product(P, q_high))
        else:
            # Roots match. Expand: (L_P x L_Q) ∪ [((L_P x H_Q) ∪ (H_P x L_Q) ∪ (H_P x H_Q)) x {p}]
            low = self.product(p_low, q_low)
            h1 = self.product(p_low, q_high)
            h2 = self.product(p_high, q_low)
            h3 = self.product(p_high, q_high)
            high = self.union(self.union(h1, h2), h3)
            res = self.get_node(p_var, low, high)
            
        self.op_cache[key] = res
        return res

    def quotient(self, P, Q):
        """
        Returns the algebraic quotient of division P / Q:
        P / Q = { r | r × Q ⊆ P } (largest family satisfying this property).
        """
        if Q == 0:
            return 0
        if Q == 1:
            return P
        if P == 0 or P == 1:
            return 0
            
        key = ('quotient', P, Q)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        q_var, q_low, q_high = self.nodes[Q]
        
        p_idx = self.var_index(P)
        q_idx = self.var_index(Q)
        
        if p_idx < q_idx:
            res = self.get_node(p_var, self.quotient(p_low, Q), self.quotient(p_high, Q))
        elif p_idx > q_idx:
            # Q is above P, but P doesn't contain q_var
            res = 0
        else:
            if q_low == 0:
                res = self.quotient(p_high, q_high)
            else:
                res = self.intersection(self.quotient(p_low, q_low), self.quotient(p_high, q_high))
            
        self.op_cache[key] = res
        return res

    def remainder(self, P, Q):
        """Returns the algebraic remainder of P / Q: P \\ ((P / Q) × Q)."""
        return self.difference(P, self.product(self.quotient(P, Q), Q))

    def change(self, P, var):
        """Toggles the presence of the variable 'var' in all sets of the family P."""
        if P == 0:
            return 0
        if P == 1:
            # Toggling var in {∅} yields {{var}}
            return self.element(var)
            
        key = ('change', P, var)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        p_idx = self.var_index(P)
        var_idx = self._register_var(var)
        
        if p_idx < var_idx:
            res = self.get_node(p_var, self.change(p_low, var), self.change(p_high, var))
        elif p_idx > var_idx:
            # var is above P (absent). Toggle means inserting it into all sets
            res = self.get_node(var, 0, P)
        else:
            # var is at the root. Swap low and high children
            res = self.get_node(var, p_high, p_low)
            
        self.op_cache[key] = res
        return res

    def subsets_with(self, P, var):
        """Returns the subset of set family P where each set contains 'var'."""
        if P == 0 or P == 1:
            return 0
            
        key = ('subsets_with', P, var)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        p_idx = self.var_index(P)
        var_idx = self._register_var(var)
        
        if p_idx < var_idx:
            res = self.get_node(p_var, self.subsets_with(p_low, var), self.subsets_with(p_high, var))
        elif p_idx > var_idx:
            # var is absent, so no sets contain it
            res = 0
        else:
            res = self.get_node(var, 0, p_high)
            
        self.op_cache[key] = res
        return res

    def subsets_without(self, P, var):
        """Returns the subset of set family P where each set does not contain 'var'."""
        if P == 0:
            return 0
        if P == 1:
            return 1
            
        key = ('subsets_without', P, var)
        if key in self.op_cache:
            return self.op_cache[key]
            
        p_var, p_low, p_high = self.nodes[P]
        p_idx = self.var_index(P)
        var_idx = self._register_var(var)
        
        if p_idx < var_idx:
            res = self.get_node(p_var, self.subsets_without(p_low, var), self.subsets_without(p_high, var))
        elif p_idx > var_idx:
            res = P
        else:
            res = p_low
            
        self.op_cache[key] = res
        return res

    def symmetric_k_n(self, vars_list, k):
        """
        Generates the ZBDD representing all combinations of choosing exactly 'k' 
        variables out of 'vars_list'. Highly optimized symmetric function builder.
        """
        if k < 0 or k > len(vars_list):
            return 0
        if k == 0 and len(vars_list) == 0:
            return 1
        
        # Sort variables according to ordering to ensure canonical construction
        sorted_vars = sorted(vars_list, key=lambda v: self._register_var(v))
        
        memo = {}
        def build(idx, rem_k):
            if rem_k < 0 or rem_k > (len(sorted_vars) - idx):
                return 0
            if rem_k == 0 and idx == len(sorted_vars):
                return 1
            if (idx, rem_k) in memo:
                return memo[(idx, rem_k)]
                
            v = sorted_vars[idx]
            # low: exclude v, high: include v
            low = build(idx + 1, rem_k)
            high = build(idx + 1, rem_k - 1)
            
            res = self.get_node(v, low, high)
            memo[(idx, rem_k)] = res
            return res
            
        return build(0, k)

    def filter_at_most_k(self, P, vars_list, k):
        """
        Filters the set family P to only include sets containing at most 'k' 
        variables from 'vars_list'. Uses memoized graph traversal.
        """
        vars_set = set(vars_list)
        memo = {}
        
        def filter_rec(node, rem_k):
            if rem_k < 0:
                return 0
            if node == 0:
                return 0
            if node == 1:
                return 1
                
            state = (node, rem_k)
            if state in memo:
                return memo[state]
                
            var, low, high = self.nodes[node]
            
            if var in vars_set:
                new_low = filter_rec(low, rem_k)
                new_high = filter_rec(high, rem_k - 1)
            else:
                new_low = filter_rec(low, rem_k)
                new_high = filter_rec(high, rem_k)
                
            res = self.get_node(var, new_low, new_high)
            memo[state] = res
            return res
            
        return filter_rec(P, k)

    # Optimization and queries
    def count_sets(self, P):
        """Returns the total number of sets in the family P."""
        memo = {}
        def count(node):
            if node == 0:
                return 0
            if node == 1:
                return 1
            if node in memo:
                return memo[node]
            _, low, high = self.nodes[node]
            res = count(low) + count(high)
            memo[node] = res
            return res
        return count(P)

    def count_nodes(self, P):
        """Counts the number of unique internal nodes in the ZBDD representing P."""
        visited = set()
        queue = deque([P])
        count = 0
        while queue:
            node = queue.popleft()
            if node in [0, 1] or node in visited:
                continue
            visited.add(node)
            count += 1
            _, low, high = self.nodes[node]
            queue.append(low)
            queue.append(high)
        return count

    def find_min_weight_set(self, P, weights):
        """
        Finds the set in P with the minimum cumulative weight in O(|P|) time.
        Weights is a dictionary mapping variables to float/int weights.
        Returns: (min_weight, set_of_variables) or (float('inf'), None) if family is empty.
        """
        memo = {}
        def search(node):
            if node == 0:
                return float('inf'), None
            if node == 1:
                return 0, set()
            if node in memo:
                return memo[node]
                
            var, low, high = self.nodes[node]
            w_low, set_low = search(low)
            w_high, set_high = search(high)
            
            w_high_total = w_high + weights.get(var, 0)
            
            if w_low <= w_high_total:
                res = (w_low, set_low)
            else:
                res = (w_high_total, set_high | {var})
                
            memo[node] = res
            return res
            
        return search(P)

    def find_max_weight_set(self, P, weights, capacity, values=None):
        """
        Finds the set in P that maximizes total value (or weight, if values is None)
        subject to weight <= capacity. Uses a memoized state search over the ZBDD graph.
        Memo key: (node_id, remaining_capacity)
        Returns: (max_value, set_of_variables) or (-inf, None) if no valid set exists.
        """
        if values is None:
            values = weights
            
        memo = {}
        def search(node, cap):
            if cap < 0:
                return float('-inf'), None
            if node == 0:
                return float('-inf'), None
            if node == 1:
                return 0, set()
                
            state = (node, cap)
            if state in memo:
                return memo[state]
                
            var, low, high = self.nodes[node]
            w_var = weights.get(var, 0)
            v_var = values.get(var, 0)
            
            # Option 1: Exclude var
            val_low, set_low = search(low, cap)
            
            # Option 2: Include var
            val_high, set_high = search(high, cap - w_var)
            if val_high != float('-inf'):
                val_high += v_var
                
            if val_low >= val_high:
                res = (val_low, set_low)
            else:
                res = (val_high, set_high | {var})
                
            memo[state] = res
            return res
            
        return search(P, capacity)

    def count_sets_by_size(self, P):
        """
        Computes the size distribution of sets in P.
        Returns: Dict of {set_size: count_of_sets}.
        """
        memo = {}
        def compute(node):
            if node == 0:
                return {}
            if node == 1:
                return {0: 1}
            if node in memo:
                return memo[node]
                
            _, low, high = self.nodes[node]
            dist_low = compute(low)
            dist_high = compute(high)
            
            res = {}
            for size, count in dist_low.items():
                res[size] = res.get(size, 0) + count
            for size, count in dist_high.items():
                # Including this variable increases size of each set in high by 1
                res[size + 1] = res.get(size + 1, 0) + count
                
            memo[node] = res
            return res
            
        return compute(P)

    def get_sets(self, P):
        """Helper to return the family of sets explicitly as a list of sets."""
        memo = {}
        def collect(node):
            if node == 0:
                return []
            if node == 1:
                return [set()]
            if node in memo:
                return memo[node]
                
            var, low, high = self.nodes[node]
            sets_low = collect(low)
            sets_high = collect(high)
            
            res = list(sets_low)
            for s in sets_high:
                res.append(s | {var})
            memo[node] = res
            return res
        return collect(P)
