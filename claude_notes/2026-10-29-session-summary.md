# Session summary: 2026-10-29 to 2026-10-30

A session with Claude Code on the spanning forest polynomial identity search. It started as a single hand-written script (`main.py` with `matrix.py` and `polynomial.py`), ported from the Mathematica code behind *A four-vertex, quadratic, spanning forest polynomial identity* (Vlasev and Yeats), and ended as a tested, modular package that finds the four-vertex identity on $K_8$ in seconds and proves it in a few minutes.

## Starting point

The script built the graph polynomial with a cofactor determinant of the edge Laplacian block matrix, found forests by setting edges of the tree polynomial to 1 and parsing `str(poly)` back into edges, and solved for the coefficients $a_{ij}$ with `sympy.linsolve` after expanding every product symbolically with the $a_{ij}$ appended to the exponent tuples. There were no tests.

## Work, in commit order

1. **Test suite.** Brute-force references (edge subsets with union-find) for trees and forest polynomials, Cayley's formula, and exact checks of known identities at random integer points. Strict `xfail` markers recorded the bugs below until each fix landed.
2. **Sign and ordering bug.** The constant term and the $a_{ij}$ terms had exponent tuples of different lengths, so they landed under different keys, and `solve_system` read them by dict position. The sign came out right by accident and depended on set iteration order. Fixed by padding correctly and summing all terms; `solve_problem` started returning its solution.
3. **Global state bugs.** `_spanning_forest_polynomials` used the global `num_vertices` (5), which emptied every forest polynomial on graphs with fewer vertices. A leftover `NUM_PARTITIONS = 2` filter dropped all one-part patterns, so problems with a one-part side had no unknowns and crashed. Coefficients that solve to nonzero constants were not counted in the summary.
4. **Forests as edge sets.** Replaced the string parsing with trees taken from exponent tuples, forests from removing $k - 1$ edges per tree, and patterns from union-find in order of first appearance. This also fixed patterns for marked vertices other than $0, \ldots, m-1$.
5. **Linear system without padding.** Built the equations directly from products in the edge variables, one per monomial, reduced by gcd and deduplicated with a set ($K_6$ search 8.2 s to 0.46 s).
6. **Spanning tree enumeration.** Replaced the cofactor determinant with a search that never closes a cycle and never disconnects the graph ($K_7$ 3.2 s to 0.09 s, $K_8$ 1.9 s). Removed `matrix.py`; the matrix-tree theorem remains as a test with sympy.
7. **Numeric search plus symbolic verification.** Equations from exact evaluations at random integer points (one per point), as in the Mathematica approach, with `verify` to confirm the whole family against the exact system ($K_7$ search 40 s to 0.95 s).
8. **Bitmask products.** Monomials without squares as bitmasks, with the product key `(a | b, a & b)` ($K_7$ verification 43.5 s and 2.1 GB to 7 s and 1.1 GB).
9. **Chunking.** Built the exact equations one group of monomials at a time, grouped by which of the first few edges they contain, so every pair of terms is still multiplied exactly once.
10. **README.**
11. **Structure.** Removed dead globals and stub files, split the code into `graph.py` and `identities.py` with marked vertices as a constructor argument, and added an `argparse` CLI.
12. **NumPy verification.** Base-3 monomial keys, so a product key is a sum and a block is one `np.add.outer`; per-column reduction; sparse rows; hash-based deduplication with an exact check and fallback ($K_8$ verification 19 min and 13.6 GB to 2.3 min and 3.7 GB).
13. **NumPy search.** Evaluation at all points at once modulo 31-bit primes, with exact values recovered by the Chinese remainder theorem, giving identical equations to exact evaluation.
14. **Bitmask trees and forests.** Trees as ints, all trees minus a choice of edges at once with NumPy, and cached forest polynomials ($K_8$ search 6.5 s to 3.2 s).
15. **Refactoring.** A `Solution` result object (no printing in the library), a `Problem` object instead of four values passed around, `equations.py` and `evaluation.py` split out of `identities.py`, `MultilinearPolynomial` (bitmask terms) as the one representation of forest polynomials, `logging` instead of a verbose flag, guards for more than 9 parts and for values too large for the evaluation primes, and `reduce` renamed to `balanced`.

## Results

- **Three marked vertices.** $\Psi \Phi^{\{1\},\{2\},\{3\}} = \Phi^{\{1\},\{2,3\}} \Phi^{\{1,2\},\{3\}} + \Phi^{\{1\},\{2,3\}} \Phi^{\{1,3\},\{2\}} + \Phi^{\{1,2\},\{3\}} \Phi^{\{1,3\},\{2\}}$, as one member of a 3-dimensional family.
- **Four marked vertices** ($P$ = `1111`, $Q$ = `1234`, balanced): 42 unknowns and an 8-dimensional affine family (8 free, 16 dependent, 18 forbidden) on $K_5$ through $K_8$ and the prism, matching the paper. Verified symbolically on $K_8$.
- **Other four-vertex shapes** (`1122`, `1112` and `1212` against `1234`) report no solution on $K_5$. This was not checked against earlier work.

## Performance now

Four-vertex search on $K_n$ with the defaults (NumPy, `chunk_edges=8`):

| Graph | Build | Search | Verify | Peak memory |
|---|---|---|---|---|
| $K_7$ | 0.09 s | 0.52 s | 1.5 s | 152 MB |
| $K_8$ | 1.2 s | about 2.5 s | 2.3 min | 3.7 GB |

## Decisions and conventions

- Monomials are the edges in the forest (the reverse of Brown and Yeats), and partitions are patterns numbered in order of first appearance, sorted.
- Identities are found numerically and confirmed symbolically, rather than expanded symbolically from the start.
- Every NumPy path has a pure Python counterpart, used beyond its size limits and as the reference in the tests.
- The user makes all commits; each step was sized to be one reviewable commit.

## Open items

- **Five or more marked vertices.** The exact `sympy.linsolve` is the bottleneck from about 625 unknowns (5 marked vertices). Solving modulo a prime, with rational reconstruction or CRT, would scale much further. An old run of `11111` × `12345` on $K_6$ (in `data_old.txt`) took about 2.9 hours and found a 320-dimensional family (320 free, 280 dependent, 25 forbidden). A run with the current code was started and stopped while still in `linsolve`; rerunning it is planned, and would show how much of the old time was the solve.
- **Verification on $K_9$** would take on the order of 9 hours at the current rate (about 840 billion term pairs). Checking the identity modulo a few primes, or using its structure, could avoid most of the products.
- **Peak memory of verification** depends on the largest chunk; choosing `chunk_edges` adaptively, or splitting large chunks, would bound it.
- **Other identity shapes**, such as mixed part sizes, would need a more general `Problem`.
