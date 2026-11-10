"""
Linear systems from exact evaluations of polynomials at random integer points,
with one equation per point. See docs/modular-solver.md.
"""

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from spanning_forests.aliases import Equation, Point
from spanning_forests.polynomial import MultilinearPolynomial, edge_indices

if TYPE_CHECKING:
    from spanning_forests.identities import Problem

# Random points have coordinates from 1 to MAX_POINT, which sets the failure bound in the README
MAX_POINT = 2**20

# Below 2**31, so that products of two residues fit in an int64
EVALUATION_PRIMES = [
    2**31 - d
    for d in (
        1, 19, 61, 69, 85, 99, 105, 151, 159, 171, 225, 249, 295, 325, 379, 399,
        411, 469, 477, 511, 525, 571, 579, 589, 595, 615, 619, 697, 699, 705, 711, 727,
    )
]


def random_points(num_points: int, num_edges: int, seed: int = 0) -> list[Point]:
    """Reproducible random points with coordinates from 1 to MAX_POINT."""
    rng = random.Random(seed)
    return [[rng.randint(1, MAX_POINT) for _ in range(num_edges)] for _ in range(num_points)]


def numeric_equations(
    problem: 'Problem',
    num_edges: int,
    extra_points: int = 10,
    seed: int = 0,
    use_numpy: bool = True,
) -> list[Equation]:
    """
    The linear system of a Problem from evaluating both sides at random
    integer points, in the (row, constant) form of equations.exact_equations.
    """
    points = random_points(len(problem.pairs) + extra_points, num_edges, seed)

    evaluate = evaluate_all_numpy if use_numpy else evaluate_all
    values_lhs = [evaluate(poly, points) for poly in (problem.phi_P, problem.phi_Q)]
    values_P = [evaluate(poly, points) for poly in problem.rhs_P.values()]
    values_Q = [evaluate(poly, points) for poly in problem.rhs_Q.values()]

    equations = []
    for i in range(len(points)):
        row = tuple(value_P[i] * value_Q[i] for value_P in values_P for value_Q in values_Q)
        constant = values_lhs[0][i] * values_lhs[1][i]
        equations.append((row, constant))
    return equations


def _terms(poly: MultilinearPolynomial) -> list[tuple[int, tuple[int, ...]]]:
    """The terms of a MultilinearPolynomial as (coefficient, edge indices) pairs."""
    return [(coefficient, edge_indices(mask)) for mask, coefficient in poly.terms.items()]


def _evaluate(terms: list[tuple[int, tuple[int, ...]]], point: Point) -> int:
    return sum(
        coefficient * math.prod(point[edge] for edge in edges) for coefficient, edges in terms
    )


def evaluate_all(poly: MultilinearPolynomial, points: list[Point]) -> list[int]:
    """The exact values of a polynomial at each point, one point at a time."""
    terms = _terms(poly)
    return [_evaluate(terms, point) for point in points]


def evaluate_all_numpy(
    poly: MultilinearPolynomial, points: list[Point], block_size: int = 2**22
) -> list[int]:
    """
    The exact values of a polynomial at each point, evaluated at all points
    at once modulo enough primes to recover them.
    """
    terms = _terms(poly)
    if not terms:
        return [0] * len(points)

    # Pad every term to the same degree with an extra coordinate equal to 1
    degree = max(len(edges) for _, edges in terms)
    num_edges = len(points[0])
    index = np.full((len(terms), degree), num_edges, dtype=np.int64)
    for term, (_, edges) in enumerate(terms):
        index[term, : len(edges)] = edges
    coordinates = np.array([[*point, 1] for point in points], dtype=np.int64)

    bound = 2 * sum(abs(coefficient) for coefficient, _ in terms) * max(map(max, points)) ** degree
    primes = []
    while math.prod(primes) <= bound:
        if len(primes) == len(EVALUATION_PRIMES):
            raise ValueError('values too large to recover from the evaluation primes')
        primes.append(EVALUATION_PRIMES[len(primes)])

    step = max(1, block_size // len(points))
    residues = []
    for p in primes:
        coeffs = np.array([coefficient % p for coefficient, _ in terms], dtype=np.int64)
        total = np.zeros(len(points), dtype=np.int64)
        for start in range(0, len(terms), step):
            block = index[start : start + step]
            values = coeffs[start : start + step] * np.ones((len(points), 1), dtype=np.int64)
            for position in range(degree):
                values = values * coordinates[:, block[:, position]] % p
            total = (total + values.sum(axis=1)) % p
        residues.append(total)

    modulus = math.prod(primes)
    exact = []
    for i in range(len(points)):
        value = crt([int(residue[i]) for residue in residues], primes)
        exact.append(value - modulus if value > modulus // 2 else value)
    return exact


def crt(residues: list[int], primes: list[int]) -> int:
    """The integer modulo the product of the primes with the given residues."""
    value, modulus = 0, 1
    for residue, p in zip(residues, primes):
        value += modulus * ((residue - value) * pow(modulus, -1, p) % p)
        modulus *= p
    return value

