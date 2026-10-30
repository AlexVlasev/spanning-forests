import random

import pytest
import sympy

from helpers import GRAPHS, assert_identity, make_solver
from identities import IdentitySolver

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
def test_chunking_does_not_change_equations(name, marked, P, Q):
    solver = make_solver(GRAPHS[name], marked)
    phi_P, phi_Q = solver.get_lhs(P, Q)
    sfps_P, sfps_Q = solver.get_rhs(P, Q, reduce=True)

    unchunked = solver.get_equations(phi_P, phi_Q, sfps_P, sfps_Q, chunk_edges=0)
    for chunk_edges in (2, 6):
        assert solver.get_equations(phi_P, phi_Q, sfps_P, sfps_Q, chunk_edges) == unchunked


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
