import math
from collections import Counter

import pytest
import sympy
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from evaluation import evaluate_all, random_points
from graph import Graph, complete_graph
from helpers import make_solver
from relations import product_terms, relation_dimension

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
    assert relation_dimension(graph, list(range(m)), [(k, l)]) == dimension


def test_shapes_are_multisets():
    graph = Graph(complete_graph(5))
    marked = list(range(5))
    expected = relation_dimension(graph, marked, [(2, 3)])
    assert relation_dimension(graph, marked, [(3, 2)]) == expected
    assert relation_dimension(graph, marked, [(2, 3), (3, 2)]) == expected


def test_ordered_search_counts_swaps_as_free_parameters():
    # 25 three-part partitions of 5 vertices, so 300 swaps of A * B and B * A
    solver = make_solver(complete_graph(6), list(range(5)))
    solution = solver.solve_problem('11111', '12345', balanced=True)
    swaps = math.comb(25, 2)
    assert solution.dimension == swaps + relation_dimension(solver.graph, solver.marked, [(3, 3)])


def test_no_products_means_no_relations():
    # Three marked vertices have no partition into four parts
    graph = Graph(complete_graph(5))
    assert relation_dimension(graph, [0, 1, 2], [(2, 4)]) == 0


def brute_force_dimension(graph, marked, shapes):
    """The dimension from expanding every product into monomials, over the rationals."""
    terms = product_terms(graph, marked, shapes)
    expansions = []
    for term in terms:
        expansion = Counter({(0,) * graph.num_edges: 1})
        for pattern in term:
            factor = graph.forest_polynomial(marked, pattern)
            next_expansion = Counter()
            for powers, c in expansion.items():
                for mask, d in factor.terms.items():
                    key = tuple(e + ((mask >> i) & 1) for i, e in enumerate(powers))
                    next_expansion[key] += c * d
            expansion = next_expansion
        expansions.append(expansion)

    monomials = sorted(set().union(*expansions))
    matrix = sympy.Matrix([[e[monomial] for monomial in monomials] for e in expansions])
    return len(terms) - DomainMatrix.from_Matrix(matrix).convert_to(QQ).rank()


@pytest.mark.parametrize(
    'm, n, shapes',
    [
        (3, 4, [(2, 2, 2)]),
        (3, 5, [(2, 2, 3), (1, 3, 3)]),
        (4, 4, [(2, 2, 3)]),
        (4, 4, [(2, 3, 3)]),
        (4, 5, [(2, 2, 3)]),
        (4, 4, [(2, 2, 2, 3)]),
        (4, 4, [(2, 2, 4), (2, 3, 3)]),
    ],
)
def test_sampling_agrees_with_expanding_every_product(m, n, shapes):
    graph = Graph(complete_graph(n))
    marked = list(range(m))
    assert relation_dimension(graph, marked, shapes) == brute_force_dimension(graph, marked, shapes)


def test_product_terms_are_distinct_and_unordered():
    graph = Graph(complete_graph(4))
    terms = product_terms(graph, [0, 1, 2], [(2, 2, 2)])
    # Multisets of size 3 from the 3 two-part partitions
    assert len(terms) == len(set(terms)) == 10
    assert all(list(term) == sorted(term) for term in terms)


def test_product_terms_have_one_factor_per_part_count():
    graph = Graph(complete_graph(5))
    # 7 two-part partitions, 6 three-part partitions and 1 four-part partition
    assert len(product_terms(graph, [0, 1, 2, 3], [(2, 3, 4)])) == 7 * 6


def test_dodgson_is_the_only_relation_mixing_these_shapes():
    # Example 2.4 of Choi and Yeats
    graph = Graph(complete_graph(5))
    assert relation_dimension(graph, [0, 1, 2], [(2, 2)]) == 0
    assert relation_dimension(graph, [0, 1, 2], [(1, 3), (2, 2)]) == 1


@pytest.mark.parametrize('factor', [1, 4])
def test_a_factor_common_to_every_product_cancels(factor):
    # There is only one partition of 4 vertices into 1 part, or into 4 parts
    graph = Graph(complete_graph(5))
    marked = [0, 1, 2, 3]
    assert relation_dimension(graph, marked, [(2, 3, factor)]) == 8


def test_mixed_shapes_must_have_matching_degrees():
    graph = Graph(complete_graph(4))
    with pytest.raises(ValueError, match='same number of factors'):
        relation_dimension(graph, [0, 1, 2], [(2, 2), (2, 3)])
    with pytest.raises(ValueError, match='same number of factors'):
        relation_dimension(graph, [0, 1, 2], [(2, 2), (1, 1, 2)])


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
