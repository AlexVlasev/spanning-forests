"""
The spaces of linear relations

    sum of c(A_1, ..., A_r) * phi_A_1 * ... * phi_A_r = 0

between products of spanning forest polynomials, where each product has a
shape: the multiset of the numbers of parts of its partitions. Products are
unordered, since the factors commute. Unlike the identities of identities.py
there is no fixed left hand side, except in Marking.expand_product, which
allows any number of factors.
"""

import itertools
import logging
import math
from collections import Counter, defaultdict, namedtuple
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, cast

import numpy as np
import sympy as sym

from spanning_forests.aliases import Equation, IntArray, Marked, Pattern, Point, Shape, Term
from spanning_forests.graph import MAX_PARTS, Graph, check_pattern
from spanning_forests.linear.equations import expansion_equations
from spanning_forests.linear.evaluation import EVALUATION_PRIMES, evaluate_all_numpy, random_points
from spanning_forests.linear.modular import reduced_row_echelon, solve_modular
from spanning_forests.solution import Solution

logger = logging.getLogger(__name__)

PRIME = EVALUATION_PRIMES[0]

RelationCounts = namedtuple('RelationCounts', 'products dimension inherited new')

type Relation = dict[Term, int]


class Marking:
    """
    A graph with marked vertices, and the linear relations between products
    of its spanning forest polynomials. Searches evaluate the products at
    extra_points more random points than unknowns, from the given seed.
    """

    def __init__(self, graph: Graph, marked: Marked, extra_points: int = 10, seed: int = 0) -> None:
        self.graph = graph
        self.marked = marked
        self.extra_points = extra_points
        self.seed = seed

    def product_terms(self, shapes: Sequence[Shape]) -> list[Term]:
        """
        Every distinct product with one of the shapes, as a tuple of patterns.
        Partitions with no spanning forests on the graph are left out.

        Shapes may be mixed, but need the same number of factors and the same
        total number of parts, since otherwise the degrees of the products could
        only agree on graphs of one size.
        """
        shapes = sorted({tuple(sorted(shape)) for shape in shapes})
        _check_shapes(shapes)

        terms = []
        for shape in shapes:
            choices = [
                itertools.combinations_with_replacement(
                    self.graph.forest_polynomials(self.marked, parts), times
                )
                for parts, times in sorted(Counter(shape).items())
            ]
            terms += [sum(choice, ()) for choice in itertools.product(*choices)]
        return terms

    def relation_dimension(self, shapes: Sequence[Shape]) -> int:
        """
        The dimension of the space of relations on the graph between all
        products with the given shapes, from product_terms.

        This is the number of products minus the rank of a matrix with one row
        per random point. The rank is computed modulo a prime, so like the random
        points, a degenerate choice could only make the dimension too large.
        """
        terms = self.product_terms(shapes)
        logger.info(f'Finding the relations between {len(terms)} products')
        if not terms:
            return 0

        _, pivots = self._echelon_form(terms)
        return len(terms) - len(pivots)

    def relation_counts(self, shapes: Sequence[Shape]) -> RelationCounts:
        """
        The RelationCounts of a space of relations on the graph: how many of its
        dimensions are inherited from fewer factors, as a relation with one
        factor fewer multiplied by any forest polynomial, and how many are new.

        Inheritance can pass through shapes outside the given ones, so inherited
        relations are built among all shapes with the same number of factors and
        total number of parts. All of the shapes must agree on those two numbers.
        Everything is computed modulo a prime, as in relation_dimension.
        """
        terms = self.product_terms(shapes)
        if not terms:
            return RelationCounts(0, 0, 0, 0)
        factors, total = len(terms[0]), sum(len(set(pattern)) for pattern in terms[0])

        ambient = self.product_terms(all_shapes(len(self.marked), factors, total))
        index = {term: i for i, term in enumerate(ambient)}
        kernel = self._kernel(terms)
        relations = np.zeros((len(kernel), len(ambient)), dtype=np.int64)
        relations[:, [index[term] for term in terms]] = kernel

        inherited, inherited_pivots = reduced_row_echelon(
            self._inherited(factors, total, index), PRIME
        )
        _, all_pivots = reduced_row_echelon(np.vstack([inherited, relations]), PRIME)
        new = len(all_pivots) - len(inherited_pivots)
        return RelationCounts(len(terms), len(kernel), len(kernel) - new, new)

    def find_relations(self, shapes: Sequence[Shape]) -> list[Relation]:
        """
        A basis of the space of relations on the graph between all products with
        the given shapes, from product_terms. Each relation is a dict mapping
        products to coprime integer coefficients, leaving out zeros, with the
        first coefficient positive.

        Each relation is exact for the equations from the random points, so like
        relation_dimension, a degenerate choice of points could only add spurious
        relations.
        """
        terms = self.product_terms(shapes)
        logger.info(f'Finding the relations between {len(terms)} products')
        if not terms:
            return []

        values = solve_modular(self._exact_equations(terms, self._points(terms)), len(terms))
        assert values is not None, 'a system with zero constants always has a solution'

        # Each free parameter of the solution gives one relation of the basis
        relations = dict()
        for term, value in zip(terms, values):
            for symbol, coefficient in value.as_coefficients_dict().items():
                if symbol.is_Symbol:
                    relations.setdefault(symbol, dict())[term] = coefficient
        ordered = sorted(relations, key=lambda symbol: int(symbol.name.split('_')[1]))
        return [_primitive(relations[symbol]) for symbol in ordered]

    def expand_product(
        self, product: Sequence[Pattern], shapes: Sequence[Shape], verify: bool = False
    ) -> Solution | None:
        """
        Finds the family of identities

            product of phi_A = sum of c(term) * (product of the phi of the term)

        over every term of product_terms for the shapes, from random evaluations,
        as a Solution keyed by those terms, or None if there is none. The product
        is a sequence of patterns, whose shape must have the same number of
        factors and total number of parts as the shapes. With verify=True the
        family is also confirmed by verify_expansion, and the result stored in
        Solution.verified.

        The unknowns are unordered products, unlike IdentitySolver.solve_problem,
        so when two factors have the same number of parts its family is larger,
        by the swaps of those factors.
        """
        product = tuple(product)
        for pattern in product:
            check_pattern(pattern, len(self.marked))
        _check_shapes([tuple(len(set(pattern)) for pattern in product), *shapes])
        terms = self.product_terms(shapes)
        logger.info(f'Solving for {len(terms)} unknowns')
        if not terms:
            return None

        equations = self._exact_equations(terms, self._points(terms), product)
        values = solve_modular(equations, len(terms))
        if values is None:
            return None

        solution = Solution(product, dict(zip(terms, values)))
        if verify:
            # Can only fail if the random points were degenerate
            solution.verified = self.verify_expansion(solution)
        return solution

    def verify_expansion(self, solution: Solution) -> bool:
        """
        Confirms exactly that every member of a family from expand_product is an
        identity on the graph, by expanding every product into monomials. Each
        coefficient is affine in the free parameters, so every equation must hold
        separately for the constant part and for each parameter.
        """
        polynomials = lambda patterns: [
            self.graph.forest_polynomial(self.marked, p) for p in patterns
        ]
        terms = list(solution.coefficients)
        logger.info(f'Verifying solution by expanding {len(terms) + 1} products')
        equations = expansion_equations(
            polynomials(solution.product),
            [polynomials(term) for term in terms],
            self.graph.num_edges,
        )

        coefficients = [_affine_parts(solution.coefficients[term]) for term in terms]
        for row, constant in equations:
            totals = defaultdict(Fraction, {1: Fraction(-constant)})
            for entry, parts in zip(row, coefficients):
                if entry:
                    for symbol, value in parts.items():
                        totals[symbol] += entry * value
            if any(totals.values()):
                return False
        return True

    def _points(self, terms: list[Term]) -> list[Point]:
        return random_points(len(terms) + self.extra_points, self.graph.num_edges, self.seed)

    def _kernel(self, terms: list[Term]) -> IntArray:
        """A basis modulo PRIME of the relations between the terms, one row per relation."""
        if not terms:
            return np.zeros((0, 0), dtype=np.int64)
        reduced, pivots = self._echelon_form(terms)

        is_pivot = set(pivots)
        free = [c for c in range(len(terms)) if c not in is_pivot]
        kernel = np.zeros((len(free), len(terms)), dtype=np.int64)
        kernel[np.arange(len(free)), free] = 1
        kernel[:, list(pivots)] = -reduced[:, free].T % PRIME
        return kernel

    def _echelon_form(self, terms: list[Term]) -> tuple[IntArray, tuple[int, ...]]:
        """The reduced rows and pivots modulo PRIME of the values of the terms at random points."""
        return reduced_row_echelon(self._product_matrix(terms, self._points(terms)), PRIME)

    def _inherited(self, factors: int, total: int, index: dict[Term, int]) -> IntArray:
        """
        The relations between the products in index that are a relation with one
        factor fewer times a forest polynomial, one row per relation and polynomial.
        """
        num_marked = len(self.marked)
        rows: list[IntArray] = [np.zeros((0, len(index)), dtype=np.int64)]
        for parts in _part_counts(num_marked):
            if factors == 1:
                break
            lower = self.product_terms(all_shapes(num_marked, factors - 1, total - parts))
            kernel = self._kernel(lower)
            if len(kernel) == 0:
                continue
            for pattern in self.graph.forest_polynomials(self.marked, parts):
                row = np.zeros((len(kernel), len(index)), dtype=np.int64)
                row[:, [index[_with_factor(term, pattern)] for term in lower]] = kernel
                rows.append(row)
        return np.vstack(rows)

    def _exact_equations(
        self, terms: list[Term], points: list[Point], product: Term | None = None
    ) -> list[Equation]:
        """
        The (row, constant) equations, with exact values of the products at each
        point. The constant is the value of the given product, and 0 without one.
        """
        patterns = _patterns([*terms, product]) if product else _patterns(terms)
        values = {
            p: evaluate_all_numpy(self.graph.forest_polynomial(self.marked, p), points)
            for p in patterns
        }
        return [
            (
                tuple(math.prod(values[p][i] for p in term) for term in terms),
                math.prod(values[p][i] for p in product) if product else 0,
            )
            for i in range(len(points))
        ]

    def _product_matrix(self, terms: list[Term], points: list[Point]) -> IntArray:
        """The (points, products) array of the values of the products modulo PRIME."""
        patterns = _patterns(terms)
        index = {pattern: i for i, pattern in enumerate(patterns)}
        values = np.array(
            [
                [
                    v % PRIME
                    for v in evaluate_all_numpy(
                        self.graph.forest_polynomial(self.marked, p), points
                    )
                ]
                for p in patterns
            ],
            dtype=np.int64,
        )

        factors = np.array([[index[pattern] for pattern in term] for term in terms])
        matrix = values[factors[:, 0]]
        for column in range(1, factors.shape[1]):
            matrix = matrix * values[factors[:, column]] % PRIME
        return matrix.T


def all_shapes(num_marked: int, factors: int, total: int) -> list[Shape]:
    """
    Every shape with this many factors and total number of parts, for
    num_marked vertices, within the MAX_PARTS parts that patterns can label.
    """
    parts = _part_counts(num_marked)
    return [s for s in itertools.combinations_with_replacement(parts, factors) if sum(s) == total]


def format_relation(relation: Relation) -> str:
    """A relation from find_relations with one term per line, as printed by the command line."""
    lines = []
    for term, coefficient in relation.items():
        product = ' * '.join(f'phi_{pattern}' for pattern in term)
        lines.append(f'    {"+" if coefficient > 0 else "-"} {abs(coefficient)} * {product}')
    return '\n'.join(lines)


def _check_shapes(shapes: Iterable[Shape]) -> None:
    if len({(len(shape), sum(shape)) for shape in shapes}) > 1:
        raise ValueError('every shape needs the same number of factors and total number of parts')


def _part_counts(num_marked: int) -> range:
    return range(1, min(num_marked, MAX_PARTS) + 1)


def _with_factor(term: Term, pattern: Pattern) -> Term:
    """A product with one more factor, sorted as in product_terms."""
    return tuple(sorted((*term, pattern), key=lambda p: (len(set(p)), p)))


def _affine_parts(coefficient: sym.Expr) -> dict[Any, Fraction]:
    """An affine sympy expression as a dict from its symbols, and 1, to Fractions."""
    parts = cast(dict[Any, sym.Rational], coefficient.as_coefficients_dict())
    return {symbol: Fraction(int(value.p), int(value.q)) for symbol, value in parts.items()}


def _primitive(relation: dict[Term, Any]) -> Relation:
    """A relation rescaled to coprime integer coefficients, with the first positive."""
    denominator = math.lcm(*(c.q for c in relation.values()))
    integers = {term: int(c * denominator) for term, c in relation.items()}
    divisor = math.gcd(*integers.values())
    if next(iter(integers.values())) < 0:
        divisor = -divisor
    return {term: c // divisor for term, c in integers.items()}


def _patterns(terms: Iterable[Term]) -> list[Pattern]:
    return sorted(set(itertools.chain.from_iterable(terms)))
