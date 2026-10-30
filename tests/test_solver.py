import random

import pytest
import sympy

from helpers import GRAPHS, assert_identity, make_solver

CASES = [
    # name, marked, P, Q, reduce, expected dimension of the solution space
    ('K5', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2, 3], '1111', '1234', True, 8),
    ('PRISM', [0, 1, 2, 3], '1111', '1234', True, 8),
]


@pytest.mark.parametrize('name, marked, P, Q, reduce, dimension', CASES)
def test_solution_family_satisfies_identity(name, marked, P, Q, reduce, dimension):
    solver = make_solver(GRAPHS[name], marked)
    solution = {pair: sympy.sympify(c) for pair, c in solver.solve_problem(P, Q, reduce).items()}

    free = set().union(*(c.free_symbols for c in solution.values()))
    assert len(free) == dimension

    # Every member of the family should be an identity, so check a few
    rng = random.Random(0)
    for _ in range(3):
        params = {x: rng.randint(-5, 5) for x in free}
        rhs = {pair: int(c.subs(params)) for pair, c in solution.items()}
        assert_identity(solver, (P, Q), rhs)
