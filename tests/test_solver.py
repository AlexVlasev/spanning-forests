import random

import numpy as np
import pytest
import sympy

from helpers import GRAPHS, adjacency, assert_identity, make_solver
from identities import IdentitySolver, _crt, _evaluate_all, _evaluate_all_numpy, _unique_runs
from polynomial import Polynomial

CASES = [
    # name, marked, P, Q, reduce, expected dimension of the solution space
    ('K4', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '112', '123', False, 0),
    ('K5', [0, 1, 2], '111', '122', False, 0),
    ('K5', [4, 0, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2, 3], '1111', '1234', True, 8),
    ('PRISM', [0, 1, 2, 3], '1111', '1234', True, 8),
]


def solve(name, marked, P, Q, reduce):
    solver = make_solver(GRAPHS[name], marked)
    solution = solver.solve_problem(P, Q, reduce)
    return solver, {pair: sympy.sympify(c) for pair, c in solution.items()}


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_solution_family_satisfies_identity(name, marked, P, Q, reduce, dimension):
    solver, solution = solve(name, marked, P, Q, reduce)

    free = set().union(*(c.free_symbols for c in solution.values()))
    assert len(free) == dimension

    # Every member of the family should be an identity, so check a few
    rng = random.Random(0)
    for _ in range(3):
        params = {x: rng.randint(-5, 5) for x in free}
        rhs = {pair: int(c.subs(params)) for pair, c in solution.items()}
        assert_identity(solver, (P, Q), rhs)


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_classification_covers_every_coefficient(name, marked, P, Q, reduce, dimension):
    _, solution = solve(name, marked, P, Q, reduce)
    classes = IdentitySolver.classify_solution(solution)

    assert sum(len(c) for c in classes.values()) == len(solution)
    assert len(classes['free']) == dimension


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_numeric_and_symbolic_systems_agree(name, marked, P, Q, reduce, dimension):
    solver = make_solver(GRAPHS[name], marked)
    phi_P, phi_Q = solver.get_lhs(P, Q)
    sfps_P, sfps_Q = solver.get_rhs(P, Q, reduce)
    a_vars, _, _ = solver.get_abstract_vars(sfps_P, sfps_Q)

    numeric = solver.get_numeric_equations(phi_P, phi_Q, sfps_P, sfps_Q)
    symbolic = solver.get_equations(phi_P, phi_Q, sfps_P, sfps_Q)
    assert solver.solve_system(numeric, a_vars) == solver.solve_system(symbolic, a_vars)


@pytest.mark.parametrize(
    'name, marked, P, Q', [('K5', [0, 1, 2], '111', '123'), ('PRISM', [0, 1, 2, 3], '1111', '1234')]
)
def test_chunking_and_numpy_do_not_change_equations(name, marked, P, Q):
    solver = make_solver(GRAPHS[name], marked)
    polys = (*solver.get_lhs(P, Q), *solver.get_rhs(P, Q, reduce=True))

    expected = solver.get_equations(*polys, chunk_edges=0, use_numpy=False)
    for use_numpy in (False, True):
        for chunk_edges in (0, 2, 6, 8):
            assert solver.get_equations(*polys, chunk_edges, use_numpy) == expected


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_numpy_evaluation_gives_the_same_numeric_equations(name, marked, P, Q, reduce, dimension):
    solver = make_solver(GRAPHS[name], marked)
    polys = (*solver.get_lhs(P, Q), *solver.get_rhs(P, Q, reduce))
    expected = solver.get_numeric_equations(*polys, use_numpy=False)
    assert solver.get_numeric_equations(*polys, use_numpy=True) == expected


def test_numpy_evaluation_with_negative_and_large_coefficients():
    variables = ['x', 'y', 'z']
    coefficients = {(1, 1, 0): -3, (0, 2, 1): 10**30, (0, 0, 0): -(10**25), (1, 0, 0): 7}
    poly = Polynomial(variables, coefficients)
    points = [[1, 2, 3], [1000, 999, 998], [5, 1, 1000]]
    assert _evaluate_all_numpy(poly, points) == _evaluate_all(poly, points)
    assert _evaluate_all_numpy(Polynomial(variables, {}), points) == [0, 0, 0]


def runs_of(columns, coeffs, runs):
    return {(tuple(columns[s : s + n]), tuple(coeffs[s : s + n])) for s, n in runs}


def test_unique_runs(monkeypatch):
    # Runs: [(0, 1), (2, 3)], [(1, 5)], [(0, 1), (2, 3)], [(0, 1), (2, 4)], [(1, 5)]
    columns = np.array([0, 2, 1, 0, 2, 0, 2, 1])
    coeffs = np.array([1, 3, 5, 1, 3, 1, 4, 5])
    starts = np.array([0, 2, 3, 5, 7])
    lengths = np.array([2, 1, 2, 2, 1])
    expected = {((0, 2), (1, 3)), ((1,), (5,)), ((0, 2), (1, 4))}

    runs = list(_unique_runs(starts, lengths, columns, coeffs))
    assert len(runs) == 3
    assert runs_of(columns, coeffs, runs) == expected

    # With every hash colliding, the exact comparison takes over
    class ZeroWeights:
        def integers(self, low, high, size, dtype):
            return np.zeros(size, dtype=dtype)

    monkeypatch.setattr(np.random, 'default_rng', lambda seed: ZeroWeights())
    runs = list(_unique_runs(starts, lengths, columns, coeffs))
    assert len(runs) == 3
    assert runs_of(columns, coeffs, runs) == expected


def test_crt():
    primes = [7, 11, 13]
    for value in (0, 1, 500, 1000):
        assert _crt([value % p for p in primes], primes) == value


def test_numpy_equations_with_coefficients_other_than_one():
    solver = make_solver(GRAPHS['K5'], [0, 1, 2])
    phi_P, phi_Q, sfps_P, sfps_Q = (*solver.get_lhs('111', '123'), *solver.get_rhs('111', '123'))
    polys = (phi_P * 2, phi_Q, sfps_P, sfps_Q)
    expected = solver.get_equations(*polys, use_numpy=False)
    assert solver.get_equations(*polys, use_numpy=True) == expected


def test_numpy_equations_with_empty_polynomials():
    # In a disconnected graph phi_P is 0, so the identity has no solution
    solver = make_solver(adjacency(5, [(0, 1), (1, 2), (3, 4)]), [0, 1, 2])
    polys = (*solver.get_lhs('111', '123'), *solver.get_rhs('111', '123', reduce=True))
    expected = solver.get_equations(*polys, use_numpy=False)
    assert solver.get_equations(*polys, use_numpy=True) == expected


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_verify_solution(name, marked, P, Q, reduce, dimension):
    solver, solution = solve(name, marked, P, Q, reduce)
    assert solver.verify_solution(P, Q, solution, reduce)

    # Changing any nonzero coefficient breaks the identity
    pair = next(pair for pair, c in solution.items() if c != 0)
    tampered = {**solution, pair: solution[pair] + 1}
    assert not solver.verify_solution(P, Q, tampered, reduce)


def test_solve_problem_with_verify():
    solver = make_solver(GRAPHS['K5'], [0, 1, 2, 3])
    assert solver.solve_problem('1111', '1234', reduce=True, verify=True) is not None


def test_classify_solution():
    x, y = sympy.symbols('x y')
    values = {'a': 0, 'b': x, 'c': 1, 'd': -2, 'e': 1 - x, 'f': 2 * y}
    classes = IdentitySolver.classify_solution(values)
    assert {kind: [key for key, _ in entries] for kind, entries in classes.items()} == {
        'forbidden': ['a'],
        'free': ['b'],
        'fixed': ['c', 'd'],
        'dependent': ['e', 'f'],
    }
