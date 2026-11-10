# Verification by expansion

A search finds a family of identities from random points. Verification proves the whole family on that graph by expanding every product into monomials, using exact arithmetic. It says nothing about other graphs.

## The linear system

Expand the left-hand side and every product on the right. Collecting the coefficient of one monomial on both sides gives one linear equation in the unknowns, so the identity holds exactly when all of these equations hold.

Each coefficient in a family is affine in the free parameters, so one check covers every member of the family. Each equation has to hold for the constant part and separately for every parameter. `IdentitySolver` does this with `sympy`, and `Marking.verify_expansion` does it with exact fractions, which is much faster for large families.

There are two builders in [`equations.py`](../spanning_forests/linear/equations.py): `exact_equations` for the two-factor identities of `IdentitySolver`, and `expansion_equations` for any number of factors. Both return the same kind of system, and they agree on two-factor products.

## Monomial keys

A forest polynomial has no squared variables, so a monomial is a bitmask of its edges. Products can have squares, so a product needs more than a bitmask.

- **Pure Python** identifies the monomial of a product of two masks $a$ and $b$ by the pair `(a | b, a & b)`: its edges, and which of those are squared.
- **NumPy** uses a single integer key, $\sum_e \text{exponent}_e \cdot b^e$. The key of a product is then the sum of the keys, so a whole array of products is one addition. The base $b$ is one more than the number of factors, so that the exponents, which are at most the number of factors, cannot carry.

The keys are stored in 64 bits, so $b$ to the power of the number of edges can be at most $2^{64}$. With two factors the base is 3, which allows 40 edges. Beyond that, `exact_equations` falls back to pure Python. `expansion_equations` raises an error instead.

## Chunking the two-factor expansion

Verifying on large graphs is limited by memory, since every distinct monomial would otherwise need a row. `exact_equations` works on one group of monomials at a time.

Take the first `chunk_edges` edges, 8 by default. Each polynomial's terms are sorted into buckets by which of those edges they contain. Let $x$ and $y$ be the parts of two masks on those edges. The monomial of their product has the pair `(x | y, x & y)` on those edges, which depends only on $x$ and $y$.

So the monomials fall into chunks labeled by that pair, with at most $3^8$ chunks, since each edge can be absent, present, or squared. Two products in different chunks cannot share a monomial, so each chunk can be solved on its own. Each chunk multiplies only the buckets that land in it. Every pair of terms is still multiplied exactly once, but the memory peak depends on the largest chunk.

Within a chunk, each column is reduced to its distinct monomials before the columns are combined.

## From columns to equations

Once each unknown has its monomials and coefficients, `_equations_from_columns` groups them by monomial into sparse rows. The left-hand side is column 0, and unknown $k$ is column $k + 1$. Each row is divided by the gcd of its entries, and duplicates are dropped, since many monomials give the same equation.

Rows are compared with a hash, so that dense rows are only built for the distinct ones. `_unique_runs` then checks that every row in a hash group really is identical, and falls back to comparing rows exactly if not. A hash collision cannot change the result.

## More than two factors

`expansion_equations` expands one product at a time. It multiplies the factors in order of increasing size, and works in blocks to bound the size of each intermediate array. It has no chunking, so it holds the expansion of every product in memory until the end, and very large cases may be out of reach.

Coefficients are added as integers. Before expanding a product, it checks that the product of $\sum |c|$ over the factors is below $2^{63}$. This bounds every coefficient and partial sum of the expansion, so the arithmetic cannot overflow. If the bound fails, it raises an error rather than risk a wrong answer, even when the true coefficients would have fit.

## What verification does not cover

Verification uses the forest polynomials from [`graph.py`](../spanning_forests/graph.py), the same as the search, so a mistake in generating them would affect both. The tests guard against this by comparing the polynomials with a brute-force enumeration of edge subsets.
