"""
Numerical checks of known quadratic identities at random integer points.

Exact integer arithmetic means a mismatch is a real failure, while agreement
at several random points makes a false positive vanishingly unlikely.
"""

import random

import pytest

from helpers import GRAPHS, evaluate, make_solver

NUM_POINTS = 5


def random_points(solver, seed):
    rng = random.Random(seed)
    for _ in range(NUM_POINTS):
        yield [rng.randint(1, 100) for _ in solver.edge_vars]


def assert_identity(solver, lhs, rhs):
    """
    Checks phi_P * phi_Q == sum of coeff * phi_B * phi_A, where lhs = (P, Q)
    and rhs maps (B, A) pattern pairs to their coefficients.
    """
    patterns = {*lhs, *(p for pair in rhs for p in pair)}
    polys = {p: solver.get_phi_P(p) for p in patterns}
    for point in random_points(solver, seed=len(solver.edge_vars)):
        value = {p: evaluate(poly, point) for p, poly in polys.items()}
        left = value[lhs[0]] * value[lhs[1]]
        right = sum(c * value[b] * value[a] for (b, a), c in rhs.items())
        assert left == right


@pytest.mark.parametrize('name', GRAPHS)
def test_three_vertex_identity(name):
    solver = make_solver(GRAPHS[name], [0, 1, 2])
    rhs = {
        ('122', '121'): 1,
        ('122', '112'): 1,
        ('121', '112'): 1,
    }
    assert_identity(solver, ('111', '123'), rhs)


# One point of the 8-dimensional solution space that the solver finds for
# P = '1111', Q = '1234' (all free parameters set to 0). The B's are the
# 2-part and the A's the 3-part partitions, as in Vlasev–Yeats.
FOUR_VERTEX_RHS = {
    ('1222', '1231'): 1,
    ('1222', '1123'): 1,
    ('1211', '1223'): 1,
    ('1211', '1123'): 1,
    ('1212', '1233'): 1,
    ('1212', '1231'): 1,
    ('1212', '1223'): 1,
    ('1212', '1123'): 1,
    ('1221', '1233'): -1,
    ('1221', '1123'): 1,
}


@pytest.mark.parametrize('name', ['K5', 'HOUSE', 'PRISM'])
def test_four_vertex_identity(name):
    solver = make_solver(GRAPHS[name], [0, 1, 2, 3])
    assert_identity(solver, ('1111', '1234'), FOUR_VERTEX_RHS)
