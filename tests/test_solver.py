import random

import pytest
import sympy

from helpers import GRAPHS, assert_identity, make_solver
from main import Solver

CASES = [
    # name, marked, P, Q, reduce, expected dimension of the solution space
    ('K4', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '112', '123', False, 0),
    ('K5', [0, 1, 2], '111', '122', False, 0),
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
    classes = Solver.classify_solution(solution)

    assert sum(len(c) for c in classes.values()) == len(solution)
    assert len(classes['free']) == dimension


def test_classify_solution():
    x, y = sympy.symbols('x y')
    values = {'a': 0, 'b': x, 'c': 1, 'd': -2, 'e': 1 - x, 'f': 2 * y}
    classes = Solver.classify_solution(values)
    assert {kind: [key for key, _ in entries] for kind, entries in classes.items()} == {
        'forbidden': ['a'],
        'free': ['b'],
        'fixed': ['c', 'd'],
        'dependent': ['e', 'f'],
    }
