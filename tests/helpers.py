"""
Independent reference implementations used to check the solver.

Everything here is brute force on purpose: enumerate edge subsets directly
instead of going through the matrix tree theorem, so that the tests do not
share any logic with the code under test.
"""

import math
import random
from itertools import combinations

from spanning_forests.graph import Graph, complete_graph
from spanning_forests.identities import IdentitySolver


def adjacency(num_vertices, edges):
    matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
    for i, j in edges:
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix


K4 = complete_graph(4)
K5 = complete_graph(5)
HOUSE = adjacency(5, [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
PRISM = adjacency(
    6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
)

GRAPHS = {'K4': K4, 'K5': K5, 'HOUSE': HOUSE, 'PRISM': PRISM}


def make_solver(matrix, marked):
    return IdentitySolver(Graph(matrix), marked)


def edge_list(matrix):
    """Edges in the same order as Graph.edges."""
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


def edges_of(mask, num_edges):
    return frozenset(e for e in range(num_edges) if (mask >> e) & 1)


def monomials(poly):
    """The edge index sets of a MultilinearPolynomial whose coefficients are all 1."""
    assert all(c == 1 for c in poly.terms.values()), poly.terms
    return {edges_of(mask, len(poly.variables)) for mask in poly.terms}


def evaluate(poly, point):
    return sum(
        c * math.prod(point[e] for e in edges_of(mask, len(poly.variables)))
        for mask, c in poly.terms.items()
    )


def random_points(graph, seed, num_points=5):
    rng = random.Random(seed)
    for _ in range(num_points):
        yield [rng.randint(1, 100) for _ in graph.edges]


def assert_identity(solver, lhs, rhs):
    """
    Checks phi_P * phi_Q == sum of coeff * phi_B * phi_A at random integer
    points, where lhs = (P, Q) and rhs maps (B, A) pattern pairs to their
    coefficients.

    Exact integer arithmetic means a mismatch is a real failure, while
    agreement at several random points makes a false positive vanishingly
    unlikely.
    """
    graph = solver.graph
    patterns = {*lhs, *(p for pair in rhs for p in pair)}
    polys = {p: graph.forest_polynomial(solver.marked, p) for p in patterns}
    for point in random_points(graph, seed=graph.num_edges):
        value = {p: evaluate(poly, point) for p, poly in polys.items()}
        left = value[lhs[0]] * value[lhs[1]]
        right = sum(c * value[b] * value[a] for (b, a), c in rhs.items())
        assert left == right
