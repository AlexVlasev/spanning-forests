import pytest

from helpers import GRAPHS, get_kn, make_solver, monomials, spanning_forests


@pytest.mark.parametrize('n', [3, 4, 5])
def test_cayley_formula(n):
    solver = make_solver(get_kn(n), [0])
    assert len(solver.tree_poly.coefficients) == n ** (n - 2)
    assert set(solver.tree_poly.coefficients.values()) == {1}


@pytest.mark.parametrize('name', GRAPHS)
def test_tree_poly_matches_brute_force(name):
    matrix = GRAPHS[name]
    solver = make_solver(matrix, [0])
    expected = {forest for forest, _ in spanning_forests(matrix, 1)}
    assert monomials(solver.tree_poly) == expected
