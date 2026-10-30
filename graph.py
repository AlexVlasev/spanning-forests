"""
Graphs, their spanning trees, and their spanning forest polynomials.

1. Enumerate the spanning trees, giving the tree polynomial and its reciprocal psi_G
2. Remove k - 1 edges from every spanning tree to get every spanning forest with k trees
3. Label each forest by the partition it induces on the marked vertices
4. Collect the forest monomials by partition to form the phi_G polynomials
"""

import itertools

import numpy as np

from polynomial import MultilinearPolynomial


def complete_graph(n):
    """The adjacency matrix of the complete graph K_n."""
    return [[0 if i == j else 1 for j in range(n)] for i in range(n)]


class Graph:
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_vertices = len(matrix)

        self.edges = []
        self.edge_vars = []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if matrix[i][j]:
                    self.edges.append((i, j))
                    self.edge_vars.append(f'e({i},{j})')
        self.num_edges = len(self.edges)

        # Each spanning tree as a bitmask, with bit e set if it contains edge e
        self.trees = self._enumerate_trees()

        # The tree polynomial has the edges in each tree, and the graph
        # polynomial psi_G the edges not in each tree
        all_edges = (1 << self.num_edges) - 1
        self.tree_poly = MultilinearPolynomial(self.edge_vars, {tree: 1 for tree in self.trees})
        self.graph_poly = MultilinearPolynomial(
            self.edge_vars, {all_edges ^ tree: 1 for tree in self.trees}
        )

        # Forest polynomials by (marked vertices, number of parts)
        self._forest_polynomials = dict()

    def _is_connected(self, excluded):
        """Whether the edges not in excluded connect every vertex."""
        neighbours = [[] for _ in range(self.num_vertices)]
        for e, (i, j) in enumerate(self.edges):
            if e not in excluded:
                neighbours[i].append(j)
                neighbours[j].append(i)

        seen = {0}
        stack = [0]
        while stack:
            for w in neighbours[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == self.num_vertices

    def _enumerate_trees(self):
        """
        Lists every spanning tree as a bitmask of its edges.

        Walks the edges in order, deciding for each whether the tree contains
        it. An edge is only included if it joins two different trees of the
        forest chosen so far, and only excluded if the remaining edges still
        connect the graph, so every branch of the search ends in a spanning
        tree.
        """
        trees = []
        excluded = set()

        def search(e, tree, size, components):
            if size == self.num_vertices - 1:
                trees.append(tree)
                return

            i, j = self.edges[e]
            ci, cj = components[i], components[j]
            if ci != cj:
                merged = [ci if c == cj else c for c in components]
                search(e + 1, tree | (1 << e), size + 1, merged)

            excluded.add(e)
            if self._is_connected(excluded):
                search(e + 1, tree, size, components)
            excluded.remove(e)

        if self._is_connected(excluded):
            search(0, 0, 0, list(range(self.num_vertices)))
        return trees

    def pattern(self, forest, marked):
        """
        Labels each marked vertex by the tree of the forest (a bitmask of its
        edges) containing it, numbering the trees in order of first
        appearance, e.g. '1213'.
        """
        parent = list(range(self.num_vertices))

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        for bit in _bits(forest):
            i, j = self.edges[bit.bit_length() - 1]
            parent[find(i)] = find(j)

        labels = dict()
        return ''.join(str(labels.setdefault(find(v), len(labels) + 1)) for v in marked)

    def forest_polynomials(self, marked, num_parts):
        """
        Computes the spanning forest polynomial of every partition of the
        marked vertices into num_parts parts, keyed by pattern. Results are
        cached, so repeated calls are free.

        Every spanning forest with k trees is a spanning tree minus k - 1 of
        its edges, so removing edges from the trees finds them all. A forest
        whose marked vertices span fewer than k trees has a tree without any
        marked vertex, so it belongs to no pattern.
        """
        key = (tuple(marked), num_parts)
        if key not in self._forest_polynomials:
            self._forest_polynomials[key] = self._compute_forest_polynomials(marked, num_parts)
        # A copy, so that callers cannot change the cached dict
        return dict(self._forest_polynomials[key])

    def _compute_forest_polynomials(self, marked, num_parts):
        if self.num_edges <= 64 and self.trees:
            forests = self._forests_numpy(num_parts)
        else:
            forests = self._forests(num_parts)

        terms = dict()
        for forest in forests:
            pattern = self.pattern(forest, marked)
            if len(set(pattern)) != num_parts:
                continue
            terms.setdefault(pattern, dict())[forest] = 1

        return {p: MultilinearPolynomial(self.edge_vars, terms[p]) for p in sorted(terms)}

    def _forests(self, num_parts):
        """The distinct bitmasks of every spanning tree minus num_parts - 1 of its edges."""
        forests = set()
        for tree in self.trees:
            for removed in itertools.combinations(_bits(tree), num_parts - 1):
                forests.add(tree ^ sum(removed))
        return forests

    def _forests_numpy(self, num_parts):
        """
        The distinct bitmasks of every spanning tree minus num_parts - 1 of its
        edges, computed for all trees at once. Needs at most 64 edges.
        """
        trees = np.array(self.trees, dtype=np.uint64)
        edges = np.arange(self.num_edges, dtype=np.uint64)
        # The n - 1 edges of each tree, as single bit masks, one row per tree
        contains = (trees[:, None] >> edges) & np.uint64(1)
        positions = np.nonzero(contains)[1].reshape(len(trees), self.num_vertices - 1)
        bits = np.uint64(1) << positions.astype(np.uint64)

        forests = []
        for removed in itertools.combinations(range(self.num_vertices - 1), num_parts - 1):
            forests.append(trees ^ bits[:, list(removed)].sum(axis=1, dtype=np.uint64))
        return [int(forest) for forest in np.unique(np.concatenate(forests))]

    def forest_polynomial(self, marked, pattern):
        """
        The spanning forest polynomial of a single partition of the marked
        vertices, which is empty if no spanning forest induces it.
        """
        assert len(pattern) == len(marked)
        if set(pattern) == {'1'}:
            return self.tree_poly
        polys = self.forest_polynomials(marked, len(set(pattern)))
        return polys.get(pattern, MultilinearPolynomial(self.edge_vars, dict()))


def _bits(mask):
    """The set bits of a bitmask, each as a bitmask of its own, from the lowest."""
    bits = []
    while mask:
        low = mask & -mask
        bits.append(low)
        mask ^= low
    return bits
