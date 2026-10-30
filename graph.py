"""
Graphs, their spanning trees, and their spanning forest polynomials.

1. Enumerate the spanning trees, giving the tree polynomial and its reciprocal psi_G
2. Remove k - 1 edges from every spanning tree to get every spanning forest with k trees
3. Label each forest by the partition it induces on the marked vertices
4. Collect the forest monomials by partition to form the phi_G polynomials
"""

import itertools

from polynomial import Polynomial


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

        # Each spanning tree as the set of indices of its edges
        self.trees = self._enumerate_trees()

        # The tree polynomial has the edges in each tree, and the graph
        # polynomial psi_G the edges not in each tree
        tree_coefficients = dict()
        graph_coefficients = dict()
        for tree in self.trees:
            tree_coefficients[tuple(1 if e in tree else 0 for e in range(self.num_edges))] = 1
            graph_coefficients[tuple(0 if e in tree else 1 for e in range(self.num_edges))] = 1
        self.tree_poly = Polynomial(self.edge_vars, tree_coefficients)
        self.graph_poly = Polynomial(self.edge_vars, graph_coefficients)

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
        Lists every spanning tree as the set of indices of its edges.

        Walks the edges in order, deciding for each whether the tree contains
        it. An edge is only included if it joins two different trees of the
        forest chosen so far, and only excluded if the remaining edges still
        connect the graph, so every branch of the search ends in a spanning
        tree.
        """
        trees = []
        excluded = set()

        def search(e, chosen, components):
            if len(chosen) == self.num_vertices - 1:
                trees.append(frozenset(chosen))
                return

            i, j = self.edges[e]
            ci, cj = components[i], components[j]
            if ci != cj:
                merged = [ci if c == cj else c for c in components]
                search(e + 1, [*chosen, e], merged)

            excluded.add(e)
            if self._is_connected(excluded):
                search(e + 1, chosen, components)
            excluded.remove(e)

        if self._is_connected(excluded):
            search(0, [], list(range(self.num_vertices)))
        return trees

    def pattern(self, forest, marked):
        """
        Labels each marked vertex by the tree of the forest containing it,
        numbering the trees in order of first appearance, e.g. '1213'.
        """
        parent = list(range(self.num_vertices))

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        for e in forest:
            i, j = self.edges[e]
            parent[find(i)] = find(j)

        labels = dict()
        return ''.join(str(labels.setdefault(find(v), len(labels) + 1)) for v in marked)

    def forest_polynomials(self, marked, num_parts):
        """
        Computes the spanning forest polynomial of every partition of the
        marked vertices into num_parts parts, keyed by pattern.

        Every spanning forest with k trees is a spanning tree minus k - 1 of
        its edges, so removing edges from the trees finds them all. A forest
        whose marked vertices span fewer than k trees has a tree without any
        marked vertex, so it belongs to no pattern.
        """
        forests = set()
        for tree in self.trees:
            for removed in itertools.combinations(tree, num_parts - 1):
                forests.add(tree.difference(removed))

        coefficients = dict()
        for forest in forests:
            pattern = self.pattern(forest, marked)
            if len(set(pattern)) != num_parts:
                continue
            power = tuple(1 if e in forest else 0 for e in range(self.num_edges))
            coefficients.setdefault(pattern, dict())[power] = 1

        return {p: Polynomial(self.edge_vars, coefficients[p]) for p in sorted(coefficients)}

    def forest_polynomial(self, marked, pattern):
        """
        The spanning forest polynomial of a single partition of the marked
        vertices, which is empty if no spanning forest induces it.
        """
        assert len(pattern) == len(marked)
        if set(pattern) == {'1'}:
            return self.tree_poly
        polys = self.forest_polynomials(marked, len(set(pattern)))
        return polys.get(pattern, Polynomial(self.edge_vars, dict()))
