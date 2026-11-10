import math

import pytest

from evaluation import evaluate_all, random_points
from graph import Graph, complete_graph
from helpers import make_solver
from relations import relation_dimension

# Table 1 of Choi and Yeats: m, n, k, l, dimension of the relations on K_n
TABLE_1 = [
    (4, 4, 2, 2, 0),
    (4, 4, 2, 3, 8),
    (4, 4, 3, 3, 0),
    (5, 5, 2, 2, 0),
    (5, 5, 2, 3, 68),
    (5, 5, 3, 3, 20),
    (5, 5, 2, 4, 15),
    (5, 5, 3, 4, 80),
    (5, 5, 4, 4, 0),
    (5, 6, 2, 2, 0),
    (5, 6, 2, 3, 64),
    (5, 6, 3, 3, 20),
    (5, 6, 2, 4, 15),
    (5, 6, 3, 4, 64),
    (5, 6, 4, 4, 0),
    (6, 6, 2, 5, 24),
    (6, 6, 5, 5, 0),
    (6, 6, 4, 5, 390),
    (6, 7, 2, 5, 24),
    (6, 7, 5, 5, 0),
    (6, 7, 4, 5, 280),
]


@pytest.mark.parametrize('m, n, k, l, dimension', TABLE_1)
def test_table_1(m, n, k, l, dimension):
    graph = Graph(complete_graph(n))
    assert relation_dimension(graph, list(range(m)), k, l) == dimension


def test_order_of_the_part_counts_does_not_matter():
    graph = Graph(complete_graph(5))
    marked = list(range(5))
    assert relation_dimension(graph, marked, 2, 3) == relation_dimension(graph, marked, 3, 2)


def test_ordered_search_counts_swaps_as_free_parameters():
    # 25 three-part partitions of 5 vertices, so 300 swaps of A * B and B * A
    solver = make_solver(complete_graph(6), list(range(5)))
    solution = solver.solve_problem('11111', '12345', balanced=True)
    swaps = math.comb(25, 2)
    assert solution.dimension == swaps + relation_dimension(solver.graph, solver.marked, 3, 3)


def test_no_products_means_no_relations():
    # Three marked vertices have no partition into four parts
    graph = Graph(complete_graph(5))
    assert relation_dimension(graph, [0, 1, 2], 2, 4) == 0


def test_relation_on_k5_that_fails_on_k6():
    # Example 2.11 of Choi and Yeats
    terms = [
        (1, '12123', '12234'),
        (1, '12123', '11234'),
        (1, '12223', '11234'),
        (-1, '11223', '12324'),
        (-1, '11223', '12234'),
        (-1, '11123', '12324'),
    ]

    def value(n):
        graph = Graph(complete_graph(n))
        point = random_points(1, graph.num_edges)
        phi = {
            p: evaluate_all(graph.forest_polynomial(list(range(5)), p), point)[0]
            for _, a, b in terms
            for p in (a, b)
        }
        return sum(c * phi[a] * phi[b] for c, a, b in terms)

    assert value(5) == 0
    assert value(6) != 0
