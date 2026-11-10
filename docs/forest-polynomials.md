# Generating forest polynomials

[`Graph`](../spanning_forests/graph.py) computes the spanning forest polynomials of a graph with marked vertices. A partition of the marked vertices is written as a pattern, as described in the README. Each monomial is stored as a bitmask, with bit $e$ set if the forest contains edge $e$.

## Spanning trees

The search walks the edges in order and decides for each one whether the tree contains it.

- An edge is included only if it joins two different components of the forest chosen so far, so no cycle is ever closed.
- An edge is excluded only if the remaining edges still connect the graph, so no branch can get stuck.

So every branch of the search ends in a spanning tree, and no time is spent on dead ends. A disconnected graph has no spanning trees, so all of its forest polynomials are empty.

The trees give the tree polynomial directly, with the edges in each tree as its monomials. The graph polynomial $\psi_G$ uses the edges not in each tree.

## From trees to forests

Every spanning forest with $k$ trees of a connected graph extends to a spanning tree, so it is a spanning tree minus $k - 1$ of its edges. Removing every choice of $k - 1$ edges from every spanning tree therefore finds all such forests. A forest usually arises from several spanning trees, so duplicates have to be removed.

Up to 64 edges, the masks fit in NumPy's 64-bit integers, and the forests for all the trees are computed at once. For each tree, the positions of its $n - 1$ edges are found, giving one row per tree. Then each choice of $k - 1$ of those positions removes those bits from every row in a single operation. Larger graphs use the same idea in pure Python.

## From forests to patterns

Each forest is labeled by the partition it induces on the marked vertices, numbering the trees in order of first appearance. A forest with $k$ trees belongs to a partition with $k$ parts only if each tree contains a marked vertex. If the marked vertices lie in fewer than $k$ of the trees, the forest belongs to no pattern and is dropped.

The polynomials for each list of marked vertices and each number of parts are computed once and cached. The polynomial with one part is the tree polynomial, so it needs no forests at all.

Patterns are checked before use. A pattern not numbered by first appearance, such as `2111`, is rejected, since no forest could ever induce it.
