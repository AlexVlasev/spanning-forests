"""
Numerical checks of known quadratic identities at random integer points.
"""

import pytest

from helpers import GRAPHS, assert_identity, make_solver


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
