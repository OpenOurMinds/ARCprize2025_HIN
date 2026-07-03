import pytest
from discrete_solver.zbdd import ZBDDManager
from discrete_solver.solver import SetCoverSolver, KnapsackSolver, NQueensSolver, make_power_set

def test_zbdd_basics():
    manager = ZBDDManager()
    
    # Terminals
    empty = manager.empty()
    base = manager.base()
    assert empty == 0
    assert base == 1
    
    # Elements
    x = manager.element('x')
    y = manager.element('y')
    assert x > 1
    assert y > 1
    assert x != y
    
    # Canonical property (unique table)
    x2 = manager.element('x')
    assert x == x2

def test_zero_suppression():
    manager = ZBDDManager()
    # If high is 0, get_node should return low
    node = manager.get_node('x', 1, 0)
    assert node == 1

def test_set_operations():
    manager = ZBDDManager()
    
    # Families:
    # A = {{x}}
    # B = {{y}}
    A = manager.element('x')
    B = manager.element('y')
    
    # Union: {{x}, {y}}
    A_union_B = manager.union(A, B)
    assert manager.count_sets(A_union_B) == 2
    assert set(frozenset(s) for s in manager.get_sets(A_union_B)) == {frozenset(['x']), frozenset(['y'])}
    
    # Intersection: empty family
    A_inter_B = manager.intersection(A, B)
    assert A_inter_B == 0
    
    # Product: {{x, y}}
    A_prod_B = manager.product(A, B)
    assert manager.count_sets(A_prod_B) == 1
    assert manager.get_sets(A_prod_B) == [{'x', 'y'}]
    
    # Difference: A \\ B = A
    diff = manager.difference(A, B)
    assert diff == A

def test_algebraic_identities():
    manager = ZBDDManager()
    x = manager.element('x')
    y = manager.element('y')
    P = manager.union(x, y)
    
    # Distributivity: P x (Q u R) = (P x Q) u (P x R)
    z = manager.element('z')
    left = manager.product(x, manager.union(y, z))
    right = manager.union(manager.product(x, y), manager.product(x, z))
    assert left == right
    
    # Double difference law: P \\ P = 0
    assert manager.difference(P, P) == 0

def test_division():
    manager = ZBDDManager()
    
    # P = {{a, b}, {a, c}, {d}}
    # Q = {{a}}
    a = manager.element('a')
    b = manager.element('b')
    c = manager.element('c')
    d = manager.element('d')
    
    P = manager.union(manager.union(manager.product(a, b), manager.product(a, c)), d)
    Q = a
    
    # P / Q = {{b}, {c}}
    quotient = manager.quotient(P, Q)
    assert set(frozenset(s) for s in manager.get_sets(quotient)) == {frozenset(['b']), frozenset(['c'])}
    
    # Remainder = P \\ ((P / Q) * Q) = {{d}}
    rem = manager.remainder(P, Q)
    assert set(frozenset(s) for s in manager.get_sets(rem)) == {frozenset(['d'])}

def test_change_and_subsets():
    manager = ZBDDManager()
    
    # P = {{a, b}, {c}}
    a = manager.element('a')
    b = manager.element('b')
    c = manager.element('c')
    P = manager.union(manager.product(a, b), c)
    
    # subsets_with P contains 'a' -> {{a, b}}
    with_a = manager.subsets_with(P, 'a')
    assert set(frozenset(s) for s in manager.get_sets(with_a)) == {frozenset(['a', 'b'])}
    
    # subsets_without P contains 'a' -> {{c}}
    without_a = manager.subsets_without(P, 'a')
    assert set(frozenset(s) for s in manager.get_sets(without_a)) == {frozenset(['c'])}
    
    # change(P, 'b') -> {{a}, {c, b}}
    changed = manager.change(P, 'b')
    assert set(frozenset(s) for s in manager.get_sets(changed)) == {frozenset(['a']), frozenset(['c', 'b'])}

def test_symmetric_and_size_count():
    manager = ZBDDManager()
    vars_list = ['x1', 'x2', 'x3', 'x4']
    
    # symmetric_k_n for k=2 should give 6 combinations
    comb_2 = manager.symmetric_k_n(vars_list, 2)
    assert manager.count_sets(comb_2) == 6
    assert manager.count_sets_by_size(comb_2) == {2: 6}
    
    # symmetric_k_n for k=3 should give 4 combinations
    comb_3 = manager.symmetric_k_n(vars_list, 3)
    assert manager.count_sets(comb_3) == 4
    assert manager.count_sets_by_size(comb_3) == {3: 4}

def test_min_weight_and_knapsack():
    manager = ZBDDManager()
    
    # P = {{a}, {b, c}, {a, b}}
    a = manager.element('a')
    b = manager.element('b')
    c = manager.element('c')
    P = manager.union(manager.union(a, manager.product(b, c)), manager.product(a, b))
    
    weights = {'a': 10, 'b': 3, 'c': 4}
    
    # Min weight set: {b, c} with weight 7 (since a is 10, and a+b is 13)
    min_w, best_set = manager.find_min_weight_set(P, weights)
    assert min_w == 7
    assert best_set == {'b', 'c'}
    
    # Knapsack solver: capacity = 8
    # max value subject to weight <= 8
    # subsets:
    # {a} weight 10 (invalid, >8)
    # {b, c} weight 7 (valid, value = weight = 7)
    # {a, b} weight 13 (invalid, >8)
    # So {b, c} is chosen
    val, selection = manager.find_max_weight_set(P, weights, 8)
    assert val == 7
    assert selection == {'b', 'c'}

def test_set_cover_solver():
    # Universe: {1, 2, 3, 4, 5}
    # Subsets:
    # S1 = {1, 2, 3} (weight 2)
    # S2 = {3, 4} (weight 1)
    # S3 = {4, 5} (weight 1)
    # S4 = {1, 5} (weight 3)
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
    res = solver.solve()
    assert res is not None
    # Best cover is S1 and S3 (covers 1, 2, 3 and 4, 5) with cost 2 + 1 = 3
    # Wait, can we choose S1 and S3? Yes, elements covered: {1, 2, 3} U {4, 5} = {1, 2, 3, 4, 5}. Cost = 3.
    # What about S2, S3, S4? Cost = 1 + 1 + 3 = 5.
    # What about S1, S2, S3? Elements covered: {1, 2, 3} U {3, 4} U {4, 5} = {1, 2, 3, 4, 5}. Cost = 2 + 1 + 1 = 4.
    # So S1 and S3 is indeed the optimal cover with cost 3!
    assert res['cost'] == 3.0
    assert res['cover'] == {'S1', 'S3'}

def test_knapsack_solver():
    items = ['item1', 'item2', 'item3', 'item4']
    weights = {'item1': 2, 'item2': 3, 'item3': 4, 'item4': 5}
    values = {'item1': 3, 'item2': 4, 'item3': 5, 'item4': 8}
    capacity = 8
    
    solver = KnapsackSolver(items, weights, values, capacity)
    res = solver.solve()
    assert res is not None
    # Combinations of weight <= 8:
    # item1 + item2 + item3 -> weight 2+3+4=9 (invalid)
    # item1 + item4 -> weight 2+5=7, value 3+8=11
    # item2 + item4 -> weight 3+5=8, value 4+8=12
    # item3 + item4 -> weight 4+5=9 (invalid)
    # item1 + item2 -> weight 5, value 7
    # So optimal is item2 + item4 with value 12 and weight 8
    assert res['value'] == 12
    assert res['selection'] == {'item2', 'item4'}
    assert res['weight'] == 8

def test_nqueens_solver():
    # 4-Queens has exactly 2 solutions
    solver_4 = NQueensSolver(4)
    res_4 = solver_4.solve()
    assert res_4['count'] == 2
    
    # 8-Queens has exactly 92 solutions
    solver_8 = NQueensSolver(8)
    res_8 = solver_8.solve()
    assert res_8['count'] == 92
