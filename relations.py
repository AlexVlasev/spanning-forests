"""
The spaces of linear relations

    sum of c(A, B) * phi_A * phi_B = 0

between products of spanning forest polynomials, where the partitions A and B
have fixed numbers of parts. There is no fixed left hand side, unlike the
identities of identities.py, and products are unordered, since
phi_A * phi_B = phi_B * phi_A.
"""

import numpy as np

from evaluation import EVALUATION_PRIMES, evaluate_all_numpy, random_points
from modular import reduced_row_echelon

PRIME = EVALUATION_PRIMES[0]


def relation_dimension(graph, marked, parts_A, parts_B, extra_points=10, seed=0):
    """
    The dimension of the space of relations on a graph, between products of
    a partition of the marked vertices into parts_A parts and one into
    parts_B parts. Partitions with no spanning forests on the graph are left
    out.

    This is the number of products minus the rank of a matrix with one row
    per random point. The rank is computed modulo a prime, so like the random
    points, a degenerate choice could only make the dimension too large.
    """
    polys_A = graph.forest_polynomials(marked, parts_A)
    polys_B = graph.forest_polynomials(marked, parts_B)
    unordered = parts_A == parts_B
    count = len(polys_A) * (len(polys_A) + 1) // 2 if unordered else len(polys_A) * len(polys_B)
    if count == 0:
        return 0

    points = random_points(count + extra_points, graph.num_edges, seed)
    values_A = _values(polys_A, points)
    values_B = _values(polys_B, points)
    _, pivots = reduced_row_echelon(_product_matrix(values_A, values_B, unordered), PRIME)
    return count - len(pivots)


def _values(polys, points):
    """The (polynomials, points) array of the values of the polynomials modulo PRIME."""
    rows = [[v % PRIME for v in evaluate_all_numpy(poly, points)] for poly in polys.values()]
    return np.array(rows, dtype=np.int64)


def _product_matrix(values_A, values_B, unordered):
    """The (points, products) array of the values of the products modulo PRIME."""
    if unordered:
        i, j = np.triu_indices(len(values_A))
        return (values_A[i] * values_B[j] % PRIME).T
    products = values_A[:, None] * values_B[None] % PRIME
    return products.reshape(-1, values_A.shape[1]).T
