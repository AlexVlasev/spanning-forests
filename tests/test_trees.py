import pytest
import sympy

from graph import Graph, complete_graph
from helpers import GRAPHS, adjacency, edge_list, monomials, spanning_forests


@pytest.mark.parametrize('n', [3, 4, 5, 6])
def test_cayley_formula(n):
    graph = Graph(complete_graph(n))
    assert len(graph.tree_poly) == n ** (n - 2)
    assert set(graph.tree_poly.terms.values()) == {1}


@pytest.mark.parametrize('name', GRAPHS)
def test_tree_poly_matches_brute_force(name):
    matrix = GRAPHS[name]
    graph = Graph(matrix)
    expected = {forest for forest, _ in spanning_forests(matrix, 1)}
    assert monomials(graph.tree_poly) == expected


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
    graph = Graph(GRAPHS[name])
    assert graph.graph_poly.to_polynomial().coefficients == matrix_tree_graph_poly(GRAPHS[name])


def test_disconnected_graph_has_no_trees():
    graph = Graph(adjacency(4, [(0, 1), (2, 3)]))
    assert graph.trees == []
    assert len(graph.tree_poly) == 0
