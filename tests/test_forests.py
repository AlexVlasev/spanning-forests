import pytest

from helpers import GRAPHS, brute_force_forests, make_solver, monomials

CASES = [
    ('K4', [0, 1, 2]),
    ('K5', [0, 1, 2]),
    ('K5', [0, 1, 2, 3]),
    ('HOUSE', [0, 1, 2]),
    ('HOUSE', [0, 1, 2, 3]),
    ('PRISM', [0, 1, 2]),
]


def assert_forest_polys_match(matrix, marked, num_trees):
    solver = make_solver(matrix, marked)
    actual = {
        pattern: monomials(poly) for pattern, poly in solver.get_sfp_for(num_trees).items()
    }
    expected = brute_force_forests(matrix, marked, num_trees)
    assert actual == expected


GLOBAL_NUM_VERTICES = pytest.mark.xfail(
    strict=True,
    reason='_spanning_forest_polynomials uses the global num_vertices (5) instead of '
    'self.num_vertices, which drops every forest on graphs with fewer vertices',
)


def forest_params():
    for name, marked in CASES:
        for k in range(2, len(marked) + 1):
            marks = GLOBAL_NUM_VERTICES if len(GRAPHS[name]) < 5 else ()
            yield pytest.param(name, marked, k, marks=marks)


@pytest.mark.parametrize('name, marked, num_trees', list(forest_params()))
def test_forest_polys_match_brute_force(name, marked, num_trees):
    assert_forest_polys_match(GRAPHS[name], marked, num_trees)


@pytest.mark.xfail(
    strict=True,
    reason='patterns label parts by the lowest vertex of each component, which only '
    'matches first-appearance order when the marked vertices are 0..m-1',
)
@pytest.mark.parametrize('name, marked', [('K5', [1, 2]), ('K5', [1, 3, 4])])
def test_forest_polys_with_arbitrary_marked_vertices(name, marked):
    assert_forest_polys_match(GRAPHS[name], marked, 2)
