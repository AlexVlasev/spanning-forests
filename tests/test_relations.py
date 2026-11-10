import math
from collections import Counter

import numpy as np
import pytest
import sympy
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from helpers import adjacency, assert_identity, make_solver
from spanning_forests.graph import Graph, complete_graph
from spanning_forests.linear.equations import _sum_by_key, exact_equations, expansion_equations
from spanning_forests.linear.evaluation import evaluate_all, random_points
from spanning_forests.relations import Marking, all_shapes, format_relation
from spanning_forests.solution import Solution

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
    assert Marking(graph, list(range(m))).relation_dimension([(k, l)]) == dimension


@pytest.mark.parametrize(
    'P, Q, n, balanced, shape',
    [
        ('1111', '1234', 5, True, (2, 3)),
        ('1111', '1234', 6, True, (2, 3)),
        ('11111', '12345', 6, False, (2, 4)),
        ('11111', '12345', 7, False, (2, 4)),
        ('1111', '1234', 7, True, (2, 3)),
    ],
)
def test_expand_product_agrees_with_solve_problem(P, Q, n, balanced, shape):
    solver = make_solver(complete_graph(n), list(range(len(P))))
    old = solver.solve_problem(P, Q, balanced=balanced)
    new = Marking(solver.graph, solver.marked).expand_product((P, Q), [shape])
    assert new.coefficients == old.coefficients


def test_expand_product_with_factors_of_more_parts_first():
    # The P_i have 4 parts and the Q_j have 3, so the factors of each term are swapped
    solver = make_solver(complete_graph(6), list(range(5)))
    old = solver.solve_problem('11223', '12134')
    new = Marking(solver.graph, solver.marked).expand_product(('11223', '12134'), [(3, 4)])
    assert new.dimension == old.dimension == 64
    zeros = {symbol: 0 for symbol in new.free_parameters}
    particular = {term: c.subs(zeros) for term, c in new.coefficients.items()}
    assert_identity(solver, ('11223', '12134'), particular)


def test_expand_product_and_solve_problem_agree_without_a_solution():
    solver = make_solver(complete_graph(6), list(range(4)))
    assert solver.solve_problem('1123', '1223') is None
    assert Marking(solver.graph, solver.marked).expand_product(('1123', '1223'), [(2, 4)]) is None


def merged_swaps(coefficients):
    """The coefficients of ordered pairs, added up over unordered products."""
    merged = {}
    for pair, c in coefficients.items():
        key = tuple(sorted(pair))
        merged[key] = merged.get(key, 0) + c
    return merged


def test_expand_product_merges_the_swaps_of_solve_problem():
    solver = make_solver(complete_graph(5), [0, 1, 2])
    old = solver.solve_problem('111', '123', balanced=True)
    new = Marking(solver.graph, solver.marked).expand_product(('111', '123'), [(2, 2)])
    assert old.dimension == new.dimension + math.comb(3, 2)
    assert merged_swaps(old.coefficients) == new.coefficients


def test_expand_product_removes_swaps_from_the_dimension():
    solver = make_solver(complete_graph(6), list(range(5)))
    old = solver.solve_problem('11111', '12345', balanced=True)
    new = Marking(solver.graph, solver.marked).expand_product(('11111', '12345'), [(3, 3)])
    assert old.dimension == new.dimension + math.comb(25, 2)


def test_expand_product_with_three_factors():
    # The spanning tree formula on the marked vertices, whose 4 triangles are missing
    graph = Graph(complete_graph(6))
    product = ('1111', '1234', '1234')
    solution = Marking(graph, [0, 1, 2, 3]).expand_product(product, [(3, 3, 3)])
    assert solution.product == product
    assert solution.dimension == 0
    assert sorted(solution.coefficients.values()).count(1) == 16
    assert sorted(solution.coefficients.values()).count(0) == 40

    total = expand(graph, [0, 1, 2, 3], product)
    for term, c in solution.coefficients.items():
        for monomial, d in expand(graph, [0, 1, 2, 3], term).items():
            total[monomial] -= int(c) * d
    assert not any(total.values())


def test_expand_product_without_a_solution():
    graph = Graph(complete_graph(5))
    assert Marking(graph, [0, 1, 2]).expand_product(('112', '121'), [(1, 3)]) is None


def test_expand_product_without_products():
    # Three marked vertices have no partition into four parts
    marking = Marking(Graph(complete_graph(5)), [0, 1, 2])
    assert marking.expand_product(('123', '123'), [(2, 4)]) is None


def test_expand_product_rejects_patterns_not_in_standard_form():
    # Three marked vertices have no four-part partitions, so there are no products to evaluate
    graph = Graph(complete_graph(5))
    with pytest.raises(ValueError, match='order of first appearance'):
        Marking(graph, [0, 1, 2]).expand_product(('211', '123'), [(2, 4)])


def test_expand_product_needs_matching_degrees():
    graph = Graph(complete_graph(4))
    with pytest.raises(ValueError, match='same number of factors'):
        Marking(graph, [0, 1, 2]).expand_product(('111', '123'), [(2, 3)])
    with pytest.raises(ValueError, match='same number of factors'):
        Marking(graph, [0, 1, 2]).expand_product(('111', '123'), [(1, 1, 2)])


@pytest.mark.parametrize(
    'n, m, product, shape, dimension',
    [
        (5, 3, ('111', '123'), (2, 2), 0),
        (6, 4, ('1111', '1234'), (2, 3), 8),
        (6, 4, ('1111', '1234', '1234'), (3, 3, 3), 0),
        (5, 4, ('1111', '1123', '1234'), (2, 3, 3), 45),
    ],
)
def test_verify_expansion(n, m, product, shape, dimension):
    graph = Graph(complete_graph(n))
    marked = list(range(m))
    solution = Marking(graph, marked).expand_product(product, [shape], verify=True)
    assert solution.dimension == dimension == Marking(graph, marked).relation_dimension([shape])
    assert solution.verified is True


def tampered(solution, change):
    """The solution with one nonzero coefficient changed by the given amount."""
    term = next(t for t, c in solution.coefficients.items() if c != 0)
    coefficients = solution.coefficients
    return Solution(solution.product, {**coefficients, term: coefficients[term] + change})


def test_verify_expansion_rejects_a_wrong_coefficient():
    graph = Graph(complete_graph(5))
    solution = Marking(graph, [0, 1, 2, 3]).expand_product(('1111', '1234'), [(2, 3)])
    assert not Marking(graph, [0, 1, 2, 3]).verify_expansion(tampered(solution, 1))


def test_verify_expansion_checks_every_member_of_the_family():
    # Correct when the new parameter is 0, and wrong for any other value
    graph = Graph(complete_graph(5))
    solution = Marking(graph, [0, 1, 2, 3]).expand_product(('1111', '1234'), [(2, 3)])
    assert not Marking(graph, [0, 1, 2, 3]).verify_expansion(tampered(solution, sympy.Symbol('y')))


def test_verify_expansion_with_rational_coefficients():
    graph = Graph(complete_graph(4))
    product = ('111', '123')
    terms = [('112', '121'), ('112', '122'), ('121', '122')]
    half = sympy.Rational(1, 2)
    good = Solution(product, {t: half + half for t in terms})
    bad = Solution(product, {**{t: sympy.Integer(1) for t in terms}, terms[0]: half})
    assert Marking(graph, [0, 1, 2]).verify_expansion(good)
    assert not Marking(graph, [0, 1, 2]).verify_expansion(bad)


def test_expansion_equations_agree_with_exact_equations():
    solver = make_solver(complete_graph(5), [0, 1, 2, 3])
    problem = solver.problem('1111', '1234', balanced=True)
    rhs_P, rhs_Q = problem.rhs_P, problem.rhs_Q
    terms = [[rhs_P[p], rhs_Q[q]] for p, q in problem.pairs]
    lhs = [problem.phi_P, problem.phi_Q]
    expected = exact_equations(problem, solver.graph.num_edges)
    assert expansion_equations(lhs, terms, solver.graph.num_edges) == expected
    assert expansion_equations(lhs, terms, solver.graph.num_edges, block_size=1) == expected


def test_sum_by_key_is_exact_beyond_the_precision_of_floats():
    keys = np.array([7, 3, 7], dtype=np.uint64)
    coeffs = np.array([2**53, 5, 1], dtype=np.int64)
    distinct, sums = _sum_by_key(keys, coeffs)
    assert distinct.tolist() == [3, 7]
    assert sums.tolist() == [5, 2**53 + 1]


def test_verify_expansion_is_exact_for_large_coefficients():
    # On K_3, phi_111 * phi_123 = phi_112 * phi_121 + phi_112 * phi_122 + phi_121 * phi_122
    # Multiplying by phi_111^38 gives coefficients well beyond the precision of floats
    graph = Graph(complete_graph(3))
    power = ('111',) * 38
    product = (*power, '111', '123')
    terms = [(*power, a, b) for a, b in [('112', '121'), ('112', '122'), ('121', '122')]]
    assert Marking(graph, [0, 1, 2]).verify_expansion(Solution(product, {t: 1 for t in terms}))
    marking = Marking(graph, [0, 1, 2])
    assert not marking.verify_expansion(Solution(product, {**{t: 1 for t in terms}, terms[0]: 2}))


def test_expansion_equations_reject_coefficients_too_large():
    # 16 monomials in each of 16 factors bounds the coefficients only by 2**64
    tree_poly = Graph(complete_graph(4)).tree_poly
    with pytest.raises(ValueError, match='coefficients too large'):
        expansion_equations([tree_poly] * 16, [[tree_poly] * 16], 6)


def test_expansion_equations_reject_keys_too_large():
    poly = Graph(complete_graph(3)).forest_polynomial([0, 1, 2], '111')
    with pytest.raises(ValueError, match='too many edges'):
        expansion_equations([poly, poly], [[poly, poly]], 41)


@pytest.mark.parametrize('settings', [{'seed': 1}, {'extra_points': 25}])
def test_marking_settings_do_not_change_the_relations(settings):
    graph = Graph(complete_graph(5))
    marked = [0, 1, 2, 3]
    expected = Marking(graph, marked).relation_dimension([(2, 3)])
    assert Marking(graph, marked, **settings).relation_dimension([(2, 3)]) == expected


def test_shapes_are_multisets():
    graph = Graph(complete_graph(5))
    marked = list(range(5))
    expected = Marking(graph, marked).relation_dimension([(2, 3)])
    assert Marking(graph, marked).relation_dimension([(3, 2)]) == expected
    assert Marking(graph, marked).relation_dimension([(2, 3), (3, 2)]) == expected


def test_ordered_search_counts_swaps_as_free_parameters():
    # 25 three-part partitions of 5 vertices, so 300 swaps of A * B and B * A
    solver = make_solver(complete_graph(6), list(range(5)))
    solution = solver.solve_problem('11111', '12345', balanced=True)
    swaps = math.comb(25, 2)
    marking = Marking(solver.graph, solver.marked)
    assert solution.dimension == swaps + marking.relation_dimension([(3, 3)])


@pytest.mark.parametrize('m, n, k, l, dimension', TABLE_1)
def test_relation_counts_agree_with_relation_dimension(m, n, k, l, dimension):
    graph = Graph(complete_graph(n))
    assert Marking(graph, list(range(m))).relation_counts([(k, l)]).dimension == dimension


def test_no_products_means_no_relations():
    # Three marked vertices have no partition into four parts
    graph = Graph(complete_graph(5))
    assert Marking(graph, [0, 1, 2]).relation_dimension([(2, 4)]) == 0


def expand(graph, marked, term):
    """A product of forest polynomials as a Counter of monomials, as tuples of powers."""
    expansion = Counter({(0,) * graph.num_edges: 1})
    for pattern in term:
        factor = graph.forest_polynomial(marked, pattern)
        next_expansion = Counter()
        for powers, c in expansion.items():
            for mask, d in factor.terms.items():
                key = tuple(e + ((mask >> i) & 1) for i, e in enumerate(powers))
                next_expansion[key] += c * d
        expansion = next_expansion
    return expansion


def brute_force_dimension(graph, marked, shapes):
    """The dimension from expanding every product into monomials, over the rationals."""
    terms = Marking(graph, marked).product_terms(shapes)
    expansions = [expand(graph, marked, term) for term in terms]

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
    marking = Marking(graph, marked)
    assert marking.relation_dimension(shapes) == brute_force_dimension(graph, marked, shapes)


@pytest.mark.parametrize(
    'm, n, shapes',
    [
        (3, 5, [(1, 3), (2, 2)]),
        (4, 5, [(2, 3)]),
        (4, 5, [(1, 4, 4), (3, 3, 3)]),
        (4, 5, [(1, 1, 4), (2, 2, 2)]),
        (4, 5, [(2, 2, 3)]),
    ],
)
def test_found_relations_are_a_basis_of_exact_relations(m, n, shapes):
    graph = Graph(complete_graph(n))
    marked = list(range(m))
    relations = Marking(graph, marked).find_relations(shapes)
    assert len(relations) == Marking(graph, marked).relation_dimension(shapes)

    terms = Marking(graph, marked).product_terms(shapes)
    assert sympy.Matrix([[r.get(t, 0) for t in terms] for r in relations]).rank() == len(relations)

    for relation in relations:
        total = Counter()
        for term, coefficient in relation.items():
            for monomial, c in expand(graph, marked, term).items():
                total[monomial] += coefficient * c
        assert not any(total.values())

        coefficients = list(relation.values())
        assert all(isinstance(c, int) and c != 0 for c in coefficients)
        assert coefficients[0] > 0
        assert math.gcd(*coefficients) == 1


def test_find_relations_gives_the_dodgson_identity():
    graph = Graph(complete_graph(5))
    assert Marking(graph, [0, 1, 2]).find_relations([(1, 3), (2, 2)]) == [
        {
            ('111', '123'): 1,
            ('112', '121'): -1,
            ('112', '122'): -1,
            ('121', '122'): -1,
        }
    ]


def test_find_relations_without_products():
    assert Marking(Graph(complete_graph(5)), [0, 1, 2]).find_relations([(2, 4)]) == []


def test_format_relation():
    relation = {('111', '123'): 1, ('112', '121'): -2}
    assert format_relation(relation) == '    + 1 * phi_111 * phi_123\n    - 2 * phi_112 * phi_121'


def rational_rank(rows):
    return DomainMatrix.from_Matrix(sympy.Matrix(rows)).convert_to(QQ).rank() if rows else 0


def exact_new_dimension(graph, marked, shapes):
    """The new relations, from exact bases of the relations, computed over the rationals."""
    target = Marking(graph, marked).find_relations(shapes)
    m, factors, total = len(marked), len(shapes[0]), sum(shapes[0])
    ambient = Marking(graph, marked).product_terms(all_shapes(m, factors, total))

    generators = []
    for parts in range(1, m + 1):
        marking = Marking(graph, marked)
        for relation in marking.find_relations(all_shapes(m, factors - 1, total - parts)):
            for pattern in graph.forest_polynomials(marked, parts):
                generators.append(
                    {
                        tuple(sorted((*t, pattern), key=lambda p: (len(set(p)), p))): c
                        for t, c in relation.items()
                    }
                )

    rows = [[r.get(term, 0) for term in ambient] for r in generators]
    both = rows + [[r.get(term, 0) for term in ambient] for r in target]
    return rational_rank(both) - rational_rank(rows)


COUNTS = [
    # m, n, shapes, dimension, inherited, new
    (3, 5, [(1, 3), (2, 2)], 1, 0, 1),
    (4, 5, [(2, 3)], 8, 0, 8),
    (4, 5, [(2, 3, 4)], 8, 8, 0),
    (4, 5, [(1, 2, 3)], 8, 8, 0),
    (4, 6, [(1, 1, 4), (2, 2, 2)], 1, 1, 0),
]


@pytest.mark.parametrize('m, n, shapes, dimension, inherited, new', COUNTS)
def test_relation_counts(m, n, shapes, dimension, inherited, new):
    graph = Graph(complete_graph(n))
    counts = Marking(graph, list(range(m))).relation_counts(shapes)
    assert counts.dimension == dimension
    assert counts.inherited == inherited
    assert counts.new == new


@pytest.mark.parametrize(
    'm, n, shapes',
    [
        (3, 4, [(1, 3), (2, 2)]),
        (3, 4, [(2, 2, 2)]),
        (3, 4, [(1, 2, 3), (2, 2, 2)]),
        (4, 5, [(2, 3, 4)]),
        (4, 5, [(1, 1, 4), (2, 2, 2)]),
        (4, 5, [(1, 4, 4), (3, 3, 3)]),
        (4, 5, [(2, 2, 3)]),
        (4, 5, [(2, 3, 3)]),
    ],
)
def test_relation_counts_agree_with_exact_bases(m, n, shapes):
    graph = Graph(complete_graph(n))
    marked = list(range(m))
    marking = Marking(graph, marked)
    assert marking.relation_counts(shapes).new == exact_new_dimension(graph, marked, shapes)


def test_inheritance_can_pass_through_shapes_outside_the_target():
    # The derivation of phi_1111^2 * phi_1234 goes through shape (1, 2, 3)
    graph = Graph(complete_graph(6))
    assert Marking(graph, [0, 1, 2, 3]).relation_counts([(1, 1, 4), (2, 2, 2)]).new == 0


def test_relation_counts_without_products():
    assert Marking(Graph(complete_graph(5)), [0, 1, 2]).relation_counts([(2, 4)]).products == 0


def test_all_shapes():
    assert all_shapes(4, 3, 9) == [(1, 4, 4), (2, 3, 4), (3, 3, 3)]
    assert all_shapes(4, 1, 5) == []


def test_all_shapes_stay_within_the_parts_that_patterns_can_label():
    assert all_shapes(10, 2, 18) == [(9, 9)]
    assert all_shapes(10, 1, 10) == []


def path_graph(n):
    return Graph(adjacency(n, [(i, i + 1) for i in range(n - 1)]))


@pytest.mark.parametrize('shape', [(9, 9), (9, 9, 9)])
def test_relation_counts_with_more_marked_vertices_than_parts(shape):
    # Lower degrees would include partitions into 10 parts
    graph = path_graph(10)
    counts = Marking(graph, list(range(10))).relation_counts([shape])
    assert counts.dimension == Marking(graph, list(range(10))).relation_dimension([shape])
    assert counts.new + counts.inherited == counts.dimension


def test_product_terms_are_distinct_and_unordered():
    graph = Graph(complete_graph(4))
    terms = Marking(graph, [0, 1, 2]).product_terms([(2, 2, 2)])
    # Multisets of size 3 from the 3 two-part partitions
    assert len(terms) == len(set(terms)) == 10
    assert all(list(term) == sorted(term) for term in terms)


def test_product_terms_have_one_factor_per_part_count():
    graph = Graph(complete_graph(5))
    # 7 two-part partitions, 6 three-part partitions and 1 four-part partition
    assert len(Marking(graph, [0, 1, 2, 3]).product_terms([(2, 3, 4)])) == 7 * 6


def test_dodgson_is_the_only_relation_mixing_these_shapes():
    # Example 2.4 of Choi and Yeats
    graph = Graph(complete_graph(5))
    assert Marking(graph, [0, 1, 2]).relation_dimension([(2, 2)]) == 0
    assert Marking(graph, [0, 1, 2]).relation_dimension([(1, 3), (2, 2)]) == 1


@pytest.mark.parametrize('factor', [1, 4])
def test_a_factor_common_to_every_product_cancels(factor):
    # There is only one partition of 4 vertices into 1 part, or into 4 parts
    graph = Graph(complete_graph(5))
    marked = [0, 1, 2, 3]
    assert Marking(graph, marked).relation_dimension([(2, 3, factor)]) == 8


def test_mixed_shapes_must_have_matching_degrees():
    graph = Graph(complete_graph(4))
    with pytest.raises(ValueError, match='same number of factors'):
        Marking(graph, [0, 1, 2]).relation_dimension([(2, 2), (2, 3)])
    with pytest.raises(ValueError, match='same number of factors'):
        Marking(graph, [0, 1, 2]).relation_dimension([(2, 2), (1, 1, 2)])


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
