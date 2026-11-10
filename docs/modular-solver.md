# The modular solver

Every search ends in a linear system over the integers. [`solve_modular`](../spanning_forests/linear/modular.py) finds its exact rational solution by row reduction modulo primes. This page explains how, and what its result guarantees.

## Why modulo primes

The equations from random points have huge integer coefficients, since each is the value of a product of forest polynomials. The solutions usually have small coefficients. Row reducing over the rationals carries the huge numbers through every step, while row reducing modulo a 31-bit prime keeps every entry in a machine integer. On the 6-vertex run in the README, this is the difference between a few seconds and several hours.

## The algorithm

A system is a collection of `(row, constant)` pairs, reduced as an augmented matrix whose last column holds the constants. Primes are taken in decreasing order, starting just below $2^{31}$, so that the product of two residues fits in 64 bits. For each prime:

1. **Reduce.** Compute the reduced row echelon form modulo $p$, with its tuple of pivot columns.
2. **Compare pivots.** The pivots over the rationals are the earliest possible, because a prime can only lose a pivot or move it later. So the lexicographically smallest pivots seen so far are the ones to trust. A prime with worse pivots is unlucky and is skipped. A prime with better pivots means all the earlier primes were unlucky, so they are discarded.
3. **Check for inconsistency.** A pivot in the constant column is the equation $0 = 1$. The system is reported to have no solution once 3 primes agree on this.
4. **Lift.** Combine the residues of each entry of the reduced matrix with the Chinese remainder theorem, and recover a fraction with rational reconstruction. A fraction $n/d$ is recovered from its residue modulo $M$ when $|n|$ and $d$ are at most $\sqrt{M/2}$. If some entry has no such fraction, another prime is needed.
5. **Check exactly.** Test the lifted solution against every original equation, using exact arithmetic. If it fails, another prime is needed.

One prime is often enough, since the entries of the reduced system tend to be small.

The result has one free parameter $x_k$ for each column $k$ without a pivot. Each pivot variable is written as a constant plus a combination of the free parameters.

## What the result guarantees

**A returned solution is the complete solution set of the given equations.** Write $n$ for the number of unknowns. The rank modulo $p$ is at most the rank over the rationals, so the family has dimension $n - \operatorname{rank}_p \ge n - \operatorname{rank}_{\mathbb{Q}}$. Every member of the family satisfies every equation, which was checked exactly, so the family lies inside the solution set. The solution set has dimension $n - \operatorname{rank}_{\mathbb{Q}}$, so the two are equal. This gives the same result as `sympy.linsolve`, which `--exact-solver` uses instead.

**"No solution" is not certified in the same way.** It relies on 3 primes agreeing. A false report would need the system to be consistent over the rationals but inconsistent modulo each of those primes, which requires every one of them to divide a denominator of the solution.

**The equations themselves may come from random points.** The guarantee above is about the system the solver was given. Whether a system from random points has the same solutions as the true identity is a separate question, with the probability bound given in the README under Limitations. A bad choice of points could only make a family too large. To remove that doubt, see [verification](verification.md).

## Evaluating at random points

[`evaluate_all_numpy`](../spanning_forests/linear/evaluation.py) computes the exact value of a polynomial at many points at once. It evaluates modulo as many primes as are needed for their product to exceed twice the largest possible absolute value, and recovers the integers with the Chinese remainder theorem. It raises an error if the values could be too large for the list of primes.

The relation dimensions of [relations](relations.md) use a single prime only, and need the rank but no solution. Since the rank modulo a prime is at most the rank over the rationals, an unlucky prime could only make a dimension too large.
