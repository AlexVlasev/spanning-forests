"""
Independent reference implementations used to check the solver.

Everything here is brute force on purpose: enumerate edge subsets directly
instead of going through the matrix tree theorem, so that the tests do not
share any logic with the code under test.
"""

import math
from itertools import combinations

from main import Solver, get_kn


def adjacency(num_vertices, edges):
    matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
    for i, j in edges:
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix


K4 = get_kn(4)
K5 = get_kn(5)
HOUSE = adjacency(5, [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
PRISM = adjacency(
    6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
)

GRAPHS = {'K4': K4, 'K5': K5, 'HOUSE': HOUSE, 'PRISM': PRISM}


def make_solver(matrix, marked):
    solver = Solver(matrix)
    solver.debug_print = False
    solver.set_vertices(marked)
    return solver


def edge_list(matrix):
    """Edges in the same order as Solver.edge_vars."""
    n = len(matrix)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if matrix[i][j]]


def spanning_forests(matrix, num_trees):
    """
    Yields (edge indices, root of each vertex) for every spanning forest
    with exactly num_trees trees.
    """
    n = len(matrix)
    edges = edge_list(matrix)
    for subset in combinations(range(len(edges)), n - num_trees):
        parent = list(range(n))

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        acyclic = True
        for e in subset:
            i, j = edges[e]
            ri, rj = find(i), find(j)
            if ri == rj:
                acyclic = False
                break
            parent[ri] = rj

        if acyclic:
            yield frozenset(subset), [find(v) for v in range(n)]


def brute_force_forests(matrix, marked, num_trees):
    """
    Maps each partition pattern of the marked vertices to the set of forests
    (as edge index sets) in its spanning forest polynomial: num_trees trees,
    each containing the marked vertices of exactly one part.

    Patterns label parts in order of first appearance, e.g. '1213'.
    """
    result = {}
    for forest, roots in spanning_forests(matrix, num_trees):
        labels = {}
        pattern = ''.join(str(labels.setdefault(roots[v], len(labels) + 1)) for v in marked)
        if len(labels) != num_trees:
            continue
        result.setdefault(pattern, set()).add(forest)
    return result


def monomials(poly):
    """The edge index sets of a polynomial whose coefficients are all 1."""
    assert all(c == 1 for c in poly.coefficients.values()), poly.coefficients
    return {frozenset(i for i, p in enumerate(power) if p) for power in poly.coefficients}


def evaluate(poly, point):
    # Solver.get_phi_P returns a plain 0 when no forest has the given pattern
    if poly == 0:
        return 0
    return sum(
        c * math.prod(x**k for x, k in zip(point, power)) for power, c in poly.coefficients.items()
    )
