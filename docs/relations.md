# Relations between products

The class [`Marking`](../spanning_forests/relations.py) holds a graph and its marked vertices, and studies the linear relations

$$\sum_{A} c_A \thinspace \Phi^{A_1} \cdots \Phi^{A_r} = 0$$

between products of spanning forest polynomials. The setup follows the spaces of spanning forest identities of Choi and Yeats ([arXiv:2610.14421](https://arxiv.org/abs/2610.14421)), extended to any number of factors.

## Products and shapes

A **product** is a tuple of patterns, one for each factor. Products are unordered, because the factors commute, so each one appears once, sorted by number of parts and then by pattern. A partition with no spanning forests on the graph is left out.

A **shape** is the multiset of the numbers of parts of the factors, written like `(2, 3, 3)`. One relation may mix several shapes, but they must have the same number of factors $r$ and the same total number of parts $t$. A forest with $k$ trees on $n$ vertices has $n - k$ edges, so a product has degree $rn - t$. Shapes with different $r$ or $t$ could only have matching degrees on a graph of one particular size. For example, the three-vertex identity in the README mixes the shapes `(1, 3)` and `(2, 2)`.

## The dimension

`relation_dimension` evaluates all $N$ products at $N$ plus a few random points, giving a matrix with one row per point and one column per product. A relation is exactly a vector in its kernel. So the dimension is $N$ minus the rank, computed modulo a single prime. A bad choice of points or an unlucky prime could only make the dimension too large.

These spaces depend on the graph. For complete graphs, Proposition 2.8 of Choi and Yeats says a relation on $K_{n+1}$ also holds on $K_n$, so the dimension can only fall as $n$ grows. The option `--sweep` shows this directly. A dimension on a small graph may include relations that fail on larger ones.

## Inherited and new relations

A relation among products with $r - 1$ factors can be multiplied by any forest polynomial to give a relation with $r$ factors. These are not really new, so `relation_counts` separates them out.

1. **The ambient space.** Take all products with the same $r$ and $t$, in every shape. A derivation can pass through shapes outside the ones asked about, so inherited relations have to be built in this larger space. For example, the relation between the shapes `(1, 1, 4)` and `(2, 2, 2)` on four marked vertices is only inherited through the shape `(1, 2, 3)`.
2. **The inherited space $I$.** For each number of parts $k$, take a basis of the relations among products with $r - 1$ factors and total $t - k$, and multiply each one by each forest polynomial with $k$ parts.
3. **The target space $X$.** Compute the relations for the shapes asked about, and write them in the ambient coordinates.
4. **The counts.** The new relations are counted by $\text{rank}(I + X) - \text{rank}(I)$, and the rest of the dimension is inherited.

The lower bases are the full relation spaces with $r - 1$ factors, so anything generated from any lower degree is included. Everything is computed modulo one prime. Sampling errors could in principle change the count of new relations in either direction, since they could enlarge both $I$ and $X$, with the same negligible probability as in the README.

Patterns can only label 9 parts, so with 10 or more marked vertices, relations multiplied by a polynomial with 10 or more parts are not counted as inherited.

## Explicit relations

`find_relations` returns an exact basis. It solves the system with a zero constant column using the [modular solver](modular-solver.md), then turns each free parameter into one relation, rescaled to coprime integers with the first coefficient positive.

## Expanding a fixed product

`expand_product` writes a fixed left-hand side as a combination of all products with the given shapes, using the value of the left-hand side as the constant column. The result is a `Solution`, as for `IdentitySolver.solve_problem`. With `verify=True` it is also checked by [expanding every product](verification.md).

The two searches differ in one way. `solve_problem` has one unknown per ordered pair $(P_i, Q_j)$, so when the $P_i$ and $Q_j$ have the same number of parts, $\Phi^A \Phi^B$ and $\Phi^B \Phi^A$ are different unknowns, and the family gains one dimension for each such swap. `expand_product` has one unknown per unordered product and no such swaps. When the numbers of parts differ, the two give the same family.
