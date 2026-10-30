import pytest
import sympy

from helpers import GRAPHS, adjacency, edge_list, get_kn, make_solver, monomials, spanning_forests


@pytest.mark.parametrize('n', [3, 4, 5, 6])
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


def matrix_tree_graph_poly(matrix):
    """
    The graph polynomial as the determinant of the block matrix

        | diag(e)  E^T |
        |  -E      0   |

    where E is the incidence matrix with one row removed.
    """
    edges = edge_list(matrix)
    variables = sympy.symbols(f'e:{len(edges)}')
    incidence = sympy.zeros(len(matrix), len(edges))
    for k, (i, j) in enumerate(edges):
        incidence[i, k] = 1
        incidence[j, k] = -1
    incidence = incidence[:-1, :]

    block = sympy.BlockMatrix(
        [[sympy.diag(*variables), incidence.T], [-incidence, sympy.zeros(incidence.rows)]]
    )
    det = sympy.Matrix(block).det(method='berkowitz')
    return sympy.Poly(det, *variables).as_dict()


@pytest.mark.parametrize('name', ['K4', 'HOUSE'])
def test_graph_poly_matches_matrix_tree_theorem(name):
    solver = make_solver(GRAPHS[name], [0])
    assert solver.graph_poly.coefficients == matrix_tree_graph_poly(GRAPHS[name])


def test_disconnected_graph_has_no_trees():
    solver = make_solver(adjacency(4, [(0, 1), (2, 3)]), [0])
    assert solver.trees == []
    assert solver.tree_poly.coefficients == {}
