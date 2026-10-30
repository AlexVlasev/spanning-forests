# Spanning forest polynomial identities

Searches for quadratic identities between spanning forest polynomials of graphs, of the form

$$\Phi^P \, \Phi^Q = \sum_{i,j} a_{ij} \, \Phi^{P_i} \, \Phi^{Q_j}$$

where $P$ and $Q$ are partitions of a set of marked vertices, and $P_i$, $Q_j$ range over all partitions of those vertices into a fixed number of parts.

This is a Python port of the Mathematica code behind *A four-vertex, quadratic, spanning forest polynomial identity* (Vlasev and Yeats). That code found identities numerically. This version finds them numerically too, and then confirms them symbolically.

## Background and conventions

For a graph $G$ and a partition $P$ of some of its vertices (the marked vertices), the spanning forest polynomial $\Phi_G^P$ sums over spanning forests with one tree per part of $P$, where each tree contains exactly the marked vertices of its part.

- **Monomials are the edges in the forest.** This is the reverse of Brown and Yeats, who use the edges not in the forest. With one part, $\Phi$ is the Kirchhoff (tree) polynomial, and $\psi_G$ is its reciprocal.
- **Partitions are written as patterns**, one digit per marked vertex, numbering parts in order of first appearance. With marked vertices `[0, 1, 2, 3]`, the pattern `1213` means the partition $\{0, 2\}, \{1\}, \{3\}$. Patterns are sorted, which fixes the numbering of the $a_{ij}$.
- **Partition sizes.** By default the $P_i$ have $|P| + 1$ parts and the $Q_j$ have $|Q| - 1$. With `reduce=True` they have $\lfloor (|P|+|Q|)/2 \rfloor$ and $\lceil (|P|+|Q|)/2 \rceil$ parts.

## How it works

1. Enumerate the spanning trees of the graph directly. The search never includes an edge that would close a cycle and never excludes one that would disconnect the graph, so every branch ends in a tree.
2. Get every spanning forest with $k$ trees by removing $k - 1$ edges from each tree. Group the forests by the partition they induce on the marked vertices.
3. **Find** the identities by evaluating every polynomial exactly at random integer points. Each point gives one linear equation in the $a_{ij}$. Using a few more points than unknowns, `sympy.linsolve` returns the full solution family, with probability 1.
4. **Verify** (optional) by expanding every product symbolically and checking that the whole solution family satisfies the exact linear system. Monomials without squares are stored as bitmasks, so the product of $a$ and $b$ is the key `(a | b, a & b)`. With NumPy (up to 40 edges), a monomial is instead the integer $\sum_e \text{exponent}_e \cdot 3^e$, so the key of a product is the sum of the keys.

## Usage

```bash
pipenv install --dev
pipenv run pytest
pipenv run python main.py 1111 1234 --graph K6 --reduce --verify
pipenv run python main.py --help
```

The command line takes the partitions $P$ and $Q$ as patterns, a graph as `K<n>` or an edge list like `0-1,1-2,2-0`, and optionally the marked vertices (by default the first `len(P)`). It prints the solution family and the identity in terms of the partitions.

From Python:

```python
from graph import Graph, complete_graph
from identities import IdentitySolver

solver = IdentitySolver(Graph(complete_graph(6)), marked=[0, 1, 2, 3])

solution = solver.solve_problem('1111', '1234', reduce=True, verify=True)
solution.coefficients  # maps (P_i, Q_j) pattern pairs to coefficients, with free parameters
solution.dimension     # 8
solution.verified      # True
print(solution.report())
print(solution.identity())
```

`solve_problem` returns a `Solution`, or `None` if there is no identity. `report()` lists each coefficient as free, fixed (a nonzero constant), dependent (determined by the free ones) or forbidden (always 0), along with a count of each, and `identity()` writes out the identity in terms of the partitions.

## Results so far

### Three marked vertices

The search finds a 3-dimensional family. One member is

$$\Psi \, \Phi^{\{1\},\{2\},\{3\}} = \Phi^{\{1\},\{2,3\}} \Phi^{\{1,2\},\{3\}} + \Phi^{\{1\},\{2,3\}} \Phi^{\{1,3\},\{2\}} + \Phi^{\{1,2\},\{3\}} \Phi^{\{1,3\},\{2\}}$$

It holds on every graph tested: $K_4$, $K_5$, the house graph and the triangular prism.

### Four marked vertices

For $P$ = `1111`, $Q$ = `1234` with `reduce=True`, there are 42 unknowns $a_{ij}$ (7 two-part $P_i$, the paper's $B$'s, times 6 three-part $Q_j$, the $A$'s). The solution is an **8-dimensional affine family**, matching the paper:

| Graph | Free | Dependent | Forbidden |
|---|---|---|---|
| $K_5$, $K_6$, $K_7$, $K_8$, prism | 8 | 16 | 18 |

One member (all free parameters set to 0) is in [tests/test_identities.py](tests/test_identities.py) as `FOUR_VERTEX_RHS`.

### Performance

Four-vertex search ($P$ = `1111`, $Q$ = `1234`) on $K_n$, on a Mac, with the default settings (NumPy, `chunk_edges=8`). Peak memory is cumulative across the phases.

| Graph | Spanning trees | Build | Search | Verify | Peak memory |
|---|---|---|---|---|---|
| $K_5$ | 125 | < 0.01 s | 0.29 s | 0.13 s | 85 MB |
| $K_6$ | 1,296 | 0.01 s | 0.35 s | 0.23 s | 90 MB |
| $K_7$ | 16,807 | 0.09 s | 0.52 s | 1.5 s | 152 MB |
| $K_8$ | 262,144 | 1.8 s | 3.2 s | 2.3 min | 3.7 GB |

Verification multiplies every pair of terms (3.6 billion on $K_8$). Its cost across versions:

| Version | $K_7$ verify | $K_7$ peak memory | $K_8$ verify | $K_8$ peak memory |
|---|---|---|---|---|
| Pure Python | 6.9 s | 1.1 GB | stopped at 16 GB | > 16 GB |
| Pure Python, chunked (`chunk_edges=6`) | 5.9 s | 201 MB | 19 min | 13.6 GB |
| NumPy, chunked (`chunk_edges=8`) | 1.5 s | 152 MB | 2.3 min | 3.7 GB |

Before chunking, verification held a sparse row for every distinct product monomial. Extrapolating from $K_6$ and $K_7$ (about 0.3–0.45 µs per term pair and 500 bytes per monomial), $K_8$ would have needed about 20 minutes and 75–80 GB. Chunking builds the equations for one group of monomials at a time, so the peak depends on the largest group. The NumPy version encodes monomials so that a product is a sum of keys, reduces each column of a group to distinct monomials before combining them, and keeps the rows sparse until duplicates are removed.

The search evaluates the polynomials at all random points at once, modulo a few primes, and recovers the exact values with the Chinese remainder theorem. Spanning trees and forests are stored as bitmasks of their edges, so all trees minus a choice of edges are computed at once with NumPy, and forest polynomials are cached. Together these took the $K_8$ search from 11.7 s to 3.2 s.

For comparison, the original version took about 8 s on the $K_6$ search and 40 s on $K_7$, with no separate verification.

## Project layout

- `graph.py`: the `Graph` class, with its spanning trees, tree and graph polynomials, and spanning forest polynomials.
- `identities.py`: the `IdentitySolver`, which searches for identities numerically and verifies them symbolically.
- `evaluation.py`: the linear system from exact evaluations at random points, modulo a few primes, with the Chinese remainder theorem.
- `equations.py`: the exact linear system from expanding every product, in chunks, with NumPy or pure Python.
- `solution.py`: the `Solution` class for a family of identities, with its report.
- `main.py`: the command line interface.
- `polynomial.py`: a small multivariate polynomial class.
- `tests/`: checks against brute-force forest enumeration, known identities, and the matrix-tree theorem.
