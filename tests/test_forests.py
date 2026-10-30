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


@pytest.mark.parametrize(
    'name, marked, num_trees',
    [(name, marked, k) for name, marked in CASES for k in range(1, len(marked) + 1)],
)
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
