"""
The spaces of linear relations

    sum of c(A_1, ..., A_r) * phi_A_1 * ... * phi_A_r = 0

between products of spanning forest polynomials, where each product has a
shape: the multiset of the numbers of parts of its partitions. Products are
unordered, since the factors commute. Unlike the identities of identities.py
there is no fixed left hand side, except in expand_product, which allows any
number of factors.
"""

import itertools
import logging
import math
from collections import Counter, defaultdict, namedtuple
from fractions import Fraction

import numpy as np

from equations import expansion_equations
from evaluation import EVALUATION_PRIMES, evaluate_all_numpy, random_points
from graph import MAX_PARTS, check_pattern
from modular import reduced_row_echelon, solve_modular
from solution import Solution

logger = logging.getLogger(__name__)

PRIME = EVALUATION_PRIMES[0]

RelationCounts = namedtuple('RelationCounts', 'products dimension inherited new')


def product_terms(graph, marked, shapes):
    """
    Every distinct product with one of the shapes, as a tuple of patterns.
    Partitions with no spanning forests on the graph are left out.

    Shapes may be mixed, but need the same number of factors and the same
    total number of parts, since otherwise the degrees of the products could
    only agree on graphs of one size.
    """
    shapes = sorted({tuple(sorted(shape)) for shape in shapes})
    _check_shapes(shapes)

    terms = []
    for shape in shapes:
        choices = [
            itertools.combinations_with_replacement(graph.forest_polynomials(marked, parts), times)
            for parts, times in sorted(Counter(shape).items())
        ]
        terms += [sum(choice, ()) for choice in itertools.product(*choices)]
    return terms


def _check_shapes(shapes):
    if len({(len(shape), sum(shape)) for shape in shapes}) > 1:
        raise ValueError('every shape needs the same number of factors and total number of parts')


def relation_dimension(graph, marked, shapes, extra_points=10, seed=0):
    """
    The dimension of the space of relations on a graph between all products
    with the given shapes, from product_terms.

    This is the number of products minus the rank of a matrix with one row
    per random point. The rank is computed modulo a prime, so like the random
    points, a degenerate choice could only make the dimension too large.
    """
    terms = product_terms(graph, marked, shapes)
    logger.info(f'Finding the relations between {len(terms)} products')
    if not terms:
        return 0

    points = random_points(len(terms) + extra_points, graph.num_edges, seed)
    _, pivots = reduced_row_echelon(_product_matrix(graph, marked, terms, points), PRIME)
    return len(terms) - len(pivots)


def all_shapes(num_marked, factors, total):
    """
    Every shape with this many factors and total number of parts, for
    num_marked vertices, within the MAX_PARTS parts that patterns can label.
    """
    parts = _part_counts(num_marked)
    return [s for s in itertools.combinations_with_replacement(parts, factors) if sum(s) == total]


def _part_counts(num_marked):
    return range(1, min(num_marked, MAX_PARTS) + 1)


def relation_counts(graph, marked, shapes, extra_points=10, seed=0):
    """
    The RelationCounts of a space of relations on a graph: how many of its
    dimensions are inherited from fewer factors, as a relation with one
    factor fewer multiplied by any forest polynomial, and how many are new.

    Inheritance can pass through shapes outside the given ones, so inherited
    relations are built among all shapes with the same number of factors and
    total number of parts. All of the shapes must agree on those two numbers.
    Everything is computed modulo a prime, as in relation_dimension.
    """
    terms = product_terms(graph, marked, shapes)
    if not terms:
        return RelationCounts(0, 0, 0, 0)
    factors, total = len(terms[0]), sum(len(set(pattern)) for pattern in terms[0])

    ambient = product_terms(graph, marked, all_shapes(len(marked), factors, total))
    index = {term: i for i, term in enumerate(ambient)}
    kernel = _kernel(graph, marked, terms, extra_points, seed)
    relations = np.zeros((len(kernel), len(ambient)), dtype=np.int64)
    relations[:, [index[term] for term in terms]] = kernel

    inherited = _inherited(graph, marked, factors, total, index, extra_points, seed)
    inherited, inherited_pivots = reduced_row_echelon(inherited, PRIME)
    _, all_pivots = reduced_row_echelon(np.vstack([inherited, relations]), PRIME)
    new = len(all_pivots) - len(inherited_pivots)
    return RelationCounts(len(terms), len(kernel), len(kernel) - new, new)


def _kernel(graph, marked, terms, extra_points, seed):
    """A basis modulo PRIME of the relations between the terms, one row per relation."""
    if not terms:
        return np.zeros((0, 0), dtype=np.int64)
    points = random_points(len(terms) + extra_points, graph.num_edges, seed)
    reduced, pivots = reduced_row_echelon(_product_matrix(graph, marked, terms, points), PRIME)

    is_pivot = set(pivots)
    free = [c for c in range(len(terms)) if c not in is_pivot]
    kernel = np.zeros((len(free), len(terms)), dtype=np.int64)
    kernel[np.arange(len(free)), free] = 1
    kernel[:, list(pivots)] = -reduced[:, free].T % PRIME
    return kernel


def _inherited(graph, marked, factors, total, index, extra_points, seed):
    """
    The relations between the products in index that are a relation with one
    factor fewer times a forest polynomial, one row per relation and polynomial.
    """
    rows = [np.zeros((0, len(index)), dtype=np.int64)]
    for parts in _part_counts(len(marked)):
        if factors == 1:
            break
        lower = product_terms(graph, marked, all_shapes(len(marked), factors - 1, total - parts))
        kernel = _kernel(graph, marked, lower, extra_points, seed)
        if len(kernel) == 0:
            continue
        for pattern in graph.forest_polynomials(marked, parts):
            row = np.zeros((len(kernel), len(index)), dtype=np.int64)
            row[:, [index[_with_factor(term, pattern)] for term in lower]] = kernel
            rows.append(row)
    return np.vstack(rows)


def _with_factor(term, pattern):
    """A product with one more factor, sorted as in product_terms."""
    return tuple(sorted((*term, pattern), key=lambda p: (len(set(p)), p)))


def find_relations(graph, marked, shapes, extra_points=10, seed=0):
    """
    A basis of the space of relations on a graph between all products with
    the given shapes, from product_terms. Each relation is a dict mapping
    products to coprime integer coefficients, leaving out zeros, with the
    first coefficient positive.

    Each relation is exact for the equations from the random points, so like
    relation_dimension, a degenerate choice of points could only add spurious
    relations.
    """
    terms = product_terms(graph, marked, shapes)
    logger.info(f'Finding the relations between {len(terms)} products')
    if not terms:
        return []

    points = random_points(len(terms) + extra_points, graph.num_edges, seed)
    values = solve_modular(_exact_equations(graph, marked, terms, points), len(terms))

    # Each free parameter of the solution gives one relation of the basis
    relations = dict()
    for term, value in zip(terms, values):
        for symbol, coefficient in value.as_coefficients_dict().items():
            if symbol.is_Symbol:
                relations.setdefault(symbol, dict())[term] = coefficient
    ordered = sorted(relations, key=lambda symbol: int(symbol.name.split('_')[1]))
    return [_primitive(relations[symbol]) for symbol in ordered]


def expand_product(graph, marked, product, shapes, verify=False, extra_points=10, seed=0):
    """
    Finds the family of identities

        product of phi_A = sum of c(term) * (product of the phi of the term)

    over every term of product_terms for the shapes, from random evaluations,
    as a Solution keyed by those terms, or None if there is none. The product
    is a sequence of patterns, whose shape must have the same number of
    factors and total number of parts as the shapes. With verify=True the
    family is also confirmed by verify_expansion, and the result stored in
    Solution.verified.

    The unknowns are unordered products, unlike IdentitySolver.solve_problem,
    so when two factors have the same number of parts its family is larger,
    by the swaps of those factors.
    """
    product = tuple(product)
    for pattern in product:
        check_pattern(pattern, len(marked))
    _check_shapes([tuple(len(set(pattern)) for pattern in product), *shapes])
    terms = product_terms(graph, marked, shapes)
    logger.info(f'Solving for {len(terms)} unknowns')
    if not terms:
        return None

    points = random_points(len(terms) + extra_points, graph.num_edges, seed)
    values = solve_modular(_exact_equations(graph, marked, terms, points, product), len(terms))
    if values is None:
        return None

    solution = Solution(product, dict(zip(terms, values)))
    if verify:
        # Can only fail if the random points were degenerate
        solution.verified = verify_expansion(graph, marked, solution)
    return solution


def verify_expansion(graph, marked, solution):
    """
    Confirms exactly that every member of a family from expand_product is an
    identity on the graph, by expanding every product into monomials. Each
    coefficient is affine in the free parameters, so every equation must hold
    separately for the constant part and for each parameter.
    """
    polynomials = lambda patterns: [graph.forest_polynomial(marked, p) for p in patterns]
    terms = list(solution.coefficients)
    logger.info(f'Verifying solution by expanding {len(terms) + 1} products')
    equations = expansion_equations(
        polynomials(solution.product), [polynomials(term) for term in terms], graph.num_edges
    )

    coefficients = [_affine_parts(solution.coefficients[term]) for term in terms]
    for row, constant in equations:
        totals = defaultdict(Fraction, {1: Fraction(-constant)})
        for entry, parts in zip(row, coefficients):
            if entry:
                for symbol, value in parts.items():
                    totals[symbol] += entry * value
        if any(totals.values()):
            return False
    return True


def _affine_parts(coefficient):
    """An affine sympy expression as a dict from its symbols, and 1, to Fractions."""
    return {
        symbol: Fraction(int(value.p), int(value.q))
        for symbol, value in coefficient.as_coefficients_dict().items()
    }


def format_relation(relation):
    """A relation from find_relations with one term per line, as printed by the command line."""
    lines = []
    for term, coefficient in relation.items():
        product = ' * '.join(f'phi_{pattern}' for pattern in term)
        lines.append(f'    {"+" if coefficient > 0 else "-"} {abs(coefficient)} * {product}')
    return '\n'.join(lines)


def _primitive(relation):
    """A relation rescaled to coprime integer coefficients, with the first positive."""
    denominator = math.lcm(*(c.q for c in relation.values()))
    integers = {term: int(c * denominator) for term, c in relation.items()}
    divisor = math.gcd(*integers.values())
    if next(iter(integers.values())) < 0:
        divisor = -divisor
    return {term: c // divisor for term, c in integers.items()}


def _patterns(terms):
    return sorted(set(itertools.chain.from_iterable(terms)))


def _exact_equations(graph, marked, terms, points, product=None):
    """
    The (row, constant) equations, with exact values of the products at each
    point. The constant is the value of the given product, and 0 without one.
    """
    patterns = _patterns([*terms, product]) if product else _patterns(terms)
    values = {p: evaluate_all_numpy(graph.forest_polynomial(marked, p), points) for p in patterns}
    return [
        (
            tuple(math.prod(values[p][i] for p in term) for term in terms),
            math.prod(values[p][i] for p in product) if product else 0,
        )
        for i in range(len(points))
    ]


def _product_matrix(graph, marked, terms, points):
    """The (points, products) array of the values of the products modulo PRIME."""
    patterns = _patterns(terms)
    index = {pattern: i for i, pattern in enumerate(patterns)}
    values = np.array(
        [
            [v % PRIME for v in evaluate_all_numpy(graph.forest_polynomial(marked, p), points)]
            for p in patterns
        ],
        dtype=np.int64,
    )

    factors = np.array([[index[pattern] for pattern in term] for term in terms])
    matrix = values[factors[:, 0]]
    for column in range(1, factors.shape[1]):
        matrix = matrix * values[factors[:, column]] % PRIME
    return matrix.T
