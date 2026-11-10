"""
Solves linear systems by row reduction modulo primes, recovering the exact
rational solution with the Chinese remainder theorem and rational
reconstruction. Much faster than solving over the rationals, since the
equations from evaluation.py have very large coefficients but their solutions
usually have small ones.
"""

import logging
import math
from collections.abc import Collection, Iterator
from fractions import Fraction

import numpy as np
import sympy as sym

from spanning_forests.aliases import Equation, IntArray
from spanning_forests.linear.evaluation import crt

logger = logging.getLogger(__name__)

# How many primes must agree before reporting that a system has no solution
NO_SOLUTION_PRIMES = 3

type Pivots = tuple[int, ...]
type Entries = dict[int, list[tuple[int, Fraction]]]


def solve_modular(equations: Collection[Equation], num_vars: int) -> tuple[sym.Expr, ...] | None:
    """
    The general solution of a linear system of (row, constant) equations, as
    a tuple of sympy expressions in free parameters x_0, x_1, ..., or None if
    there is no solution. The same result as solving over the rationals with
    sympy.linsolve.

    A returned solution is checked exactly against the equations. The rank
    modulo a prime is at most the rank over the rationals, so a family of
    that dimension satisfying every equation is the whole solution set. That
    there is no solution relies on NO_SOLUTION_PRIMES primes agreeing.
    """
    best, primes, residues = None, [], []
    for p in _primes():
        reduced, pivots = reduced_row_echelon(_residues(equations, num_vars, p), p)
        if best is None or _pivot_order(pivots) < _pivot_order(best):
            best, primes, residues = pivots, [], []
        elif pivots != best:
            # An unlucky prime, which loses or moves a pivot
            continue
        primes.append(p)
        residues.append(reduced)

        # A pivot in the constant column is an equation 0 = 1
        if num_vars in best:
            if len(primes) == NO_SOLUTION_PRIMES:
                logger.info(f'No solution modulo {len(primes)} primes')
                return None
            continue

        entries = _lift(residues, primes, num_vars, best)
        if entries is not None and _satisfies(equations, best, entries):
            logger.info(f'Solved modulo {len(primes)} prime(s)')
            return _parametrize(best, entries, num_vars)


def _primes() -> Iterator[int]:
    """The primes below 2**31 in decreasing order, so products of two residues fit in an int64."""
    p = 2**31
    while True:
        p = int(sym.prevprime(p))
        yield p


def _residues(equations: Collection[Equation], num_vars: int, p: int) -> IntArray:
    rows = [[x % p for x in row] + [constant % p] for row, constant in equations]
    return np.array(rows, dtype=np.int64).reshape(len(rows), num_vars + 1)


def _pivot_order(pivots: Pivots) -> tuple[int | float, ...]:
    """
    A sort key under which the pivots over the rationals come first: they are
    the earliest possible, so unlucky primes can only have later or fewer.
    """
    return (*pivots, math.inf)


def reduced_row_echelon(matrix: IntArray, p: int) -> tuple[IntArray, Pivots]:
    """
    The nonzero rows of the reduced row echelon form of an int64 matrix
    modulo a prime below 2**31, and the tuple of pivot columns. Changes the
    matrix in place.
    """
    pivots = []
    for column in range(matrix.shape[1]):
        rank = len(pivots)
        if rank == matrix.shape[0]:
            break
        nonzero = np.nonzero(matrix[rank:, column])[0]
        if nonzero.size == 0:
            continue
        row = rank + nonzero[0]
        matrix[[rank, row]] = matrix[[row, rank]]
        matrix[rank] = matrix[rank] * pow(int(matrix[rank, column]), -1, p) % p

        others = np.nonzero(matrix[:, column])[0]
        others = others[others != rank]
        matrix[others] = (matrix[others] - matrix[others, column, None] * matrix[rank] % p) % p
        pivots.append(column)
    return matrix[: len(pivots)], tuple(pivots)


def rational_reconstruction(value: int, modulus: int) -> Fraction | None:
    """
    The fraction n/d congruent to value modulo the modulus with |n| and d at
    most sqrt(modulus / 2), or None if there is none.
    """
    bound = math.isqrt(modulus // 2)
    r0, r1 = modulus, value % modulus
    t0, t1 = 0, 1
    while r1 > bound:
        quotient = r0 // r1
        r0, r1 = r1, r0 - quotient * r1
        t0, t1 = t1, t0 - quotient * t1
    if t1 == 0 or abs(t1) > bound or math.gcd(r1, t1) != 1:
        return None
    return Fraction(r1, t1)


def _lift(
    residues: list[IntArray], primes: list[int], num_vars: int, pivots: Pivots
) -> Entries | None:
    """
    The nonzero entries of the reduced system over the rationals, as a dict
    mapping each non-pivot column (including the constant column) to a list
    of (row, value), or None if the primes are too few to recover them.
    """
    is_pivot = set(pivots)
    columns = [c for c in range(num_vars + 1) if c not in is_pivot]
    stacked = np.stack(residues)[:, :, columns]
    modulus = math.prod(primes)

    entries = {c: [] for c in columns}
    for row, index in np.argwhere(stacked.any(axis=0)).tolist():
        value = crt([int(r) for r in stacked[:, row, index]], primes)
        fraction = rational_reconstruction(value, modulus)
        if fraction is None:
            return None
        entries[columns[index]].append((row, fraction))
    return entries


def _satisfies(equations: Collection[Equation], pivots: Pivots, entries: Entries) -> bool:
    """Whether the solution with these reduced entries satisfies every equation exactly."""
    for row, constant in equations:
        augmented = (*row, constant)
        for column, values in entries.items():
            if sum(augmented[pivots[i]] * value for i, value in values) != augmented[column]:
                return False
    return True


def _parametrize(pivots: Pivots, entries: Entries, num_vars: int) -> tuple[sym.Expr, ...]:
    xs = sym.symbols(f'x_:{num_vars}')
    terms = {pivot: [] for pivot in pivots}
    for column, values in entries.items():
        for i, value in values:
            value = sym.Rational(value.numerator, value.denominator)
            terms[pivots[i]].append(value if column == num_vars else -value * xs[column])
    return tuple(sym.Add(*terms[v]) if v in terms else xs[v] for v in range(num_vars))
