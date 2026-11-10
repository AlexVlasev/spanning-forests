# Spanning forest polynomial identities

Searches for quadratic identities between spanning forest polynomials of graphs, of the form

$$\Phi^P \thinspace \Phi^Q = \sum_{i,j} a_{ij} \thinspace \Phi^{P_i} \thinspace \Phi^{Q_j}$$

where $P$ and $Q$ are partitions of a set of marked vertices, and $P_i$, $Q_j$ range over all partitions of those vertices into a fixed number of parts.

This is a Python port of the Mathematica code behind (Vlasev and Yeats, 2012[^vlasev2012]). That code found identities numerically. This version finds them numerically too, and then confirms them symbolically.

## Background and conventions

For a graph $G$ and a partition $P$ of some of its vertices (the marked vertices), the spanning forest polynomial $\Phi_G^P$ sums over spanning forests with one tree per part of $P$, where each tree contains exactly the marked vertices of its part.

- **Monomials are the edges in the forest.** This is the reverse of Brown and Yeats, who use the edges not in the forest. With one part, $\Phi$ is the Kirchhoff (tree) polynomial, and $\psi_G$ is its reciprocal.
- **Partitions are written as patterns**, one digit per marked vertex, numbering parts in order of first appearance. With marked vertices `[0, 1, 2, 3]`, the pattern `1213` means the partition $\lbrace 0, 2\rbrace, \lbrace 1\rbrace, \lbrace 3\rbrace$. Patterns are sorted, which fixes the numbering of the $a_{ij}$.
- **Partition sizes.** By default the $P_i$ have $|P| + 1$ parts and the $Q_j$ have $|Q| - 1$. With `balanced=True` they have $\lfloor (|P|+|Q|)/2 \rfloor$ and $\lceil (|P|+|Q|)/2 \rceil$ parts.

## How it works

1. Enumerate the spanning trees of the graph directly. The search never includes an edge that would close a cycle and never excludes one that would disconnect the graph, so every branch ends in a tree.
2. Get every spanning forest with $k$ trees by removing $k - 1$ edges from each tree. Group the forests by the partition they induce on the marked vertices.
3. **Find** the identities by evaluating every polynomial exactly at random integer points. Each point gives one linear equation in the $a_{ij}$. Using a few more points than unknowns, row reduction modulo primes gives the full solution family, with probability 1. Its rational coefficients are recovered with the Chinese remainder theorem and rational reconstruction, then checked exactly against the equations. `--exact-solver` (or `modular=False`) uses `sympy.linsolve` instead, which gives the same result much more slowly.
4. **Verify** (optional) by expanding every product symbolically and checking that the whole solution family satisfies the exact linear system. Monomials without squares are stored as bitmasks, so the product of $a$ and $b$ is the key `(a | b, a & b)`. With NumPy (up to 40 edges), a monomial is instead the integer $\sum_e \text{exponent}_e \cdot 3^e$, so the key of a product is the sum of the keys.

## Usage

```bash
pipenv install --dev
pipenv run pytest
pipenv run python main.py 1111 1234 --graph K6 --balanced --verify
pipenv run python main.py --relations 2,2,3 --marked 0,1,2,3 --graph K6
pipenv run python main.py --help
```

The command line takes the partitions $P$ and $Q$ as patterns, a graph as `K<n>` or an edge list like `0-1,1-2,2-0`, and optionally the marked vertices (by default the first `len(P)`). It prints the solution family and the identity in terms of the partitions.

With `--relations`, it instead takes one or more shapes and prints the dimension of the space of linear relations between all products of forest polynomials with those shapes. A shape lists the number of parts of each factor, so `2,2,3` is a product of three forest polynomials. Mixed shapes need the same number of factors and the same total number of parts. `--show-relations` also prints a basis of the relations, with coprime integer coefficients.

From Python:

```python
from graph import Graph, complete_graph
from identities import IdentitySolver

solver = IdentitySolver(Graph(complete_graph(6)), marked=[0, 1, 2, 3])

solution = solver.solve_problem('1111', '1234', balanced=True, verify=True)
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

$$\Psi \thinspace \Phi^{\lbrace 1\rbrace,\lbrace 2\rbrace,\lbrace 3\rbrace} = \Phi^{\lbrace 1\rbrace,\lbrace 2,3\rbrace} \Phi^{\lbrace 1,2\rbrace,\lbrace 3\rbrace} + \Phi^{\lbrace 1\rbrace,\lbrace 2,3\rbrace} \Phi^{\lbrace 1,3\rbrace,\lbrace 2\rbrace} + \Phi^{\lbrace 1,2\rbrace,\lbrace 3\rbrace} \Phi^{\lbrace 1,3\rbrace,\lbrace 2\rbrace}$$

It holds on every graph tested: $K_4$, $K_5$, the house graph and the triangular prism.

### Four marked vertices

For $P$ = `1111`, $Q$ = `1234` with `balanced=True`, there are 42 unknowns $a_{ij}$ (7 two-part $P_i$, the paper's $B$'s, times 6 three-part $Q_j$, the $A$'s). The solution is an **8-dimensional affine family**, matching the paper:

| Graph | Free | Dependent | Forbidden |
|---|---|---|---|
| $K_5$, $K_6$, $K_7$, $K_8$, prism | 8 | 16 | 18 |

One member (all free parameters set to 0) is in [tests/test_identities.py](tests/test_identities.py) as `FOUR_VERTEX_RHS`.

### Five, six and seven marked vertices

For $P = 1\ldots1$ and $Q = 12\ldots m$ with the default partition sizes (two-part $P_i$ and $(m-1)$-part $Q_j$), the search finds:

| Marked vertices | Graph | Unknowns | Free | Dependent | Forbidden | Time | Time with `--exact-solver` |
|---|---|---|---|---|---|---|---|
| 5 | $K_7$ | 150 | 15 | 65 | 70 | 0.34 s | 65 s |
| 6 | $K_7$ | 465 | 24 | 216 | 225 | 1.5 s | 3.3 hours |
| 7 | $K_8$ | 1,323 | 35 | 637 | 651 | 40 s | not run |

These runs were numerical only, without symbolic verification.

### The general pattern

The numbers of free parameters for 3, 4, 5, 6 and 7 marked vertices are 3, 8, 15, 24 and 35, which is $m(m-2)$ for $m$ marked vertices. This has been proved for identities of this form, with two-part and $(m-1)$-part partitions on the right-hand side (Fraser and Yeats, 2023[^fraser2023], Theorem 3.4.1).

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

## Limitations

### Of the code

- **Shape of the identities.** Only identities $\Phi^P \Phi^Q = \sum c_{ij} \Phi^{P_i} \Phi^{Q_j}$ are searched, where all the $P_i$ have the same number of parts, and so do all the $Q_j$. The two choices of sizes are the default ($|P| + 1$ and $|Q| - 1$) and `balanced`. Other shapes, such as mixed sizes or products of three polynomials, would need changes to `IdentitySolver.problem`.
- **At most 9 parts.** Patterns label parts with single digits, so partitions with more than 9 parts are rejected. This limits the number of parts, not the number of vertices.
- **Simple graphs.** Graphs are 0/1 adjacency matrices: undirected, without multiple edges, loops or edge weights. A disconnected graph has no spanning trees, so all its forest polynomials are empty.
- **The search is probabilistic.** The random points use a fixed seed, so results are reproducible, and a wrong result would need a vanishingly unlikely degenerate choice of points. Use `--verify` to prove a family symbolically.
- **NumPy paths have size limits.** Generating forests with NumPy needs at most 64 edges, and verification at most 40 ($K_9$ has 36). Larger graphs fall back to much slower pure Python. Evaluation recovers values up to about $2^{490}$ from 16 primes, and raises an error beyond that.
- **Single process.** Everything runs in one process and in memory. With `--exact-solver`, `sympy.linsolve` is fine for tens of unknowns but slow for hundreds.

### In practice

The sizes involved grow combinatorially, both in the graph and in the number of marked vertices.

- **Spanning trees.** All spanning trees are enumerated, and $K_n$ has $n^{n-2}$: 262,144 for $K_8$, 4.8 million for $K_9$ and 100 million for $K_{10}$. Forests are generated from the trees, so $K_9$ is roughly the largest complete graph within reach, while sparser graphs with more vertices can go further.
- **Verification.** It multiplies every pair of terms. For four marked vertices on $K_n$ that is about $26.7 \thinspace n^{2n-7}$ pairs: 22 million for $K_7$, 3.6 billion for $K_8$ and about 840 billion for $K_9$. At the $K_8$ rate, verifying on $K_9$ would take on the order of 9 hours.
- **Unknowns.** For $m$ marked vertices, the number of partitions into $k$ parts is the Stirling number $S(m, k)$, and there is one unknown per pair $(P_i, Q_j)$. For $P = 1\ldots1$ and $Q = 12\ldots m$ with the default partition sizes, that is 9 unknowns for 3 vertices, 42 for 4, 150 for 5, 465 for 6 and 1,323 for 7. With `balanced`, it is the same for 3 and 4, and then 625 for 5, 5,850 for 6 and 122,500 for 7. The search needs one random point per unknown. Row reduction modulo primes is dense, with memory quadratic and time cubic in the number of unknowns, so hundreds of thousands of unknowns are out of reach.

## Project layout

- `graph.py`: the `Graph` class, with its spanning trees, tree and graph polynomials, and spanning forest polynomials.
- `identities.py`: the `IdentitySolver`, which searches for identities numerically and verifies them symbolically.
- `evaluation.py`: the linear system from exact evaluations at random points, modulo a few primes, with the Chinese remainder theorem.
- `equations.py`: the exact linear system from expanding every product, in chunks, with NumPy or pure Python.
- `modular.py`: solves linear systems by row reduction modulo primes, recovering the exact rational solution.
- `relations.py`: the dimension of the space of linear relations between products of forest polynomials with fixed numbers of parts, inspired by the spaces of spanning forest identities in (Choi and Yeats, 2026[^choi2026]).
- `solution.py`: the `Solution` class for a family of identities, with its report.
- `main.py`: the command line interface.
- `polynomial.py`: `MultilinearPolynomial`, which stores each monomial as a bitmask of its edges and is used for all spanning forest polynomials, and a general `Polynomial` class for display.
- `tests/`: checks against brute-force forest enumeration, known identities, and the matrix-tree theorem.

## References

[^vlasev2012]: Vlasev, A., & Yeats, K. (2012). A four-vertex, quadratic, spanning forest polynomial identity. *Electronic Journal of Linear Algebra*, 23(1), 923-941. 
  [![arXiv](https://img.shields.io/badge/arXiv-1106.2869-b31b1b.svg)](https://arxiv.org/abs/1106.2869) [![DOI](https://img.shields.io/badge/DOI-10.13001%2F1081--3810.1566-blue.svg)](https://doi.org/10.13001/1081-3810.1566)

[^fraser2023]: Fraser, M., & Yeats, K. (2023). Column expansion identities and quadratic spanning forest identities. *Australasian Journal of Combinatorics*, 86, 271-307. 
  [![arXiv](https://img.shields.io/badge/arXiv-2109.13401-b31b1b.svg)](https://arxiv.org/abs/2109.13401) [![AJC](https://img.shields.io/badge/AJC-86-blue.svg)](https://ajc.maths.uq.edu.au/v86.p271)

[^choi2026]: Choi, K., & Yeats, R. (2026). A framework for spanning forest identities. *arXiv preprint*. 
  [![arXiv](https://img.shields.io/badge/arXiv-2610.14421-b31b1b.svg)](https://arxiv.org/abs/2610.14421)
