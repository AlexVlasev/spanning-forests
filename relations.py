"""
The spaces of linear relations

    sum of c(A_1, ..., A_r) * phi_A_1 * ... * phi_A_r = 0

between products of spanning forest polynomials, where each product has a
shape: the multiset of the numbers of parts of its partitions. There is no
fixed left hand side, unlike the identities of identities.py, and products
are unordered, since the factors commute.
"""

import itertools
import logging
from collections import Counter

import numpy as np

from evaluation import EVALUATION_PRIMES, evaluate_all_numpy, random_points
from modular import reduced_row_echelon

logger = logging.getLogger(__name__)

PRIME = EVALUATION_PRIMES[0]


def product_terms(graph, marked, shapes):
    """
    Every distinct product with one of the shapes, as a tuple of patterns.
    Partitions with no spanning forests on the graph are left out.

    Shapes may be mixed, but need the same number of factors and the same
    total number of parts, since otherwise the degrees of the products could
    only agree on graphs of one size.
    """
    shapes = sorted({tuple(sorted(shape)) for shape in shapes})
    if len({(len(shape), sum(shape)) for shape in shapes}) > 1:
        raise ValueError('every shape needs the same number of factors and total number of parts')

    terms = []
    for shape in shapes:
        choices = [
            itertools.combinations_with_replacement(graph.forest_polynomials(marked, parts), times)
            for parts, times in sorted(Counter(shape).items())
        ]
        terms += [sum(choice, ()) for choice in itertools.product(*choices)]
    return terms


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


def _product_matrix(graph, marked, terms, points):
    """The (points, products) array of the values of the products modulo PRIME."""
    patterns = sorted(set(itertools.chain.from_iterable(terms)))
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
