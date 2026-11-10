"""
The linear relations between products of spanning forest polynomials, with
any number of factors. See docs/relations.md.
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

from spanning_forests.aliases import Equation, IntArray, Marked, Pattern, Point, Shape, Product
from spanning_forests.graph import MAX_PARTS, Graph, check_pattern
from spanning_forests.linear.equations import expansion_equations
from spanning_forests.linear.evaluation import EVALUATION_PRIMES, evaluate_all_numpy, random_points
from spanning_forests.linear.modular import reduced_row_echelon, solve_modular
from spanning_forests.solution import Solution

logger = logging.getLogger(__name__)

PRIME = EVALUATION_PRIMES[0]

RelationCounts = namedtuple('RelationCounts', 'products dimension inherited new')

type Relation = dict[Product, int]


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

    def products(self, shapes: Sequence[Shape]) -> list[Product]:
        """
        Every distinct product with one of the shapes, as a tuple of patterns.
        Shapes may be mixed, with the same number of factors and total parts.
        """
        shapes = sorted({tuple(sorted(shape)) for shape in shapes})
        _check_shapes(shapes)

        products = []
        for shape in shapes:
            choices = [
                itertools.combinations_with_replacement(
                    self.graph.forest_polynomials(self.marked, parts), times
                )
                for parts, times in sorted(Counter(shape).items())
            ]
            products += [sum(choice, ()) for choice in itertools.product(*choices)]
        return products

    def relation_dimension(self, shapes: Sequence[Shape]) -> int:
        """
        The dimension of the space of relations on the graph between all
        products with the given shapes, modulo a prime.
        """
        products = self.products(shapes)
        logger.info(f'Finding the relations between {len(products)} products')
        if not products:
            return 0

        _, pivots = self._echelon_form(products)
        return len(products) - len(pivots)

    def relation_counts(self, shapes: Sequence[Shape]) -> RelationCounts:
        """
        The RelationCounts of a space of relations on the graph: how many of its
        dimensions are inherited from fewer factors and how many are new. The
        shapes must have the same number of factors and total parts.
        """
        products = self.products(shapes)
        if not products:
            return RelationCounts(0, 0, 0, 0)
        factors, total = len(products[0]), sum(len(set(pattern)) for pattern in products[0])

        ambient = self.products(all_shapes(len(self.marked), factors, total))
        index = {product: i for i, product in enumerate(ambient)}
        kernel = self._kernel(products)
        relations = np.zeros((len(kernel), len(ambient)), dtype=np.int64)
        relations[:, [index[product] for product in products]] = kernel

        inherited, inherited_pivots = reduced_row_echelon(
            self._inherited(factors, total, index), PRIME
        )
        _, all_pivots = reduced_row_echelon(np.vstack([inherited, relations]), PRIME)
        new = len(all_pivots) - len(inherited_pivots)
        return RelationCounts(len(products), len(kernel), len(kernel) - new, new)

    def find_relations(self, shapes: Sequence[Shape]) -> list[Relation]:
        """
        A basis of the relations on the graph between all products with the given
        shapes, each a dict from products to coprime integer coefficients.
        """
        products = self.products(shapes)
        logger.info(f'Finding the relations between {len(products)} products')
        if not products:
            return []

        equations = self._exact_equations(products, self._points(products))
        values = solve_modular(equations, len(products))
        assert values is not None, 'a system with zero constants always has a solution'

        # Each free parameter of the solution gives one relation of the basis
        relations = dict()
        for product, value in zip(products, values):
            for symbol, coefficient in value.as_coefficients_dict().items():
                if symbol.is_Symbol:
                    relations.setdefault(symbol, dict())[product] = coefficient
        ordered = sorted(relations, key=lambda symbol: int(symbol.name.split('_')[1]))
        return [_primitive(relations[symbol]) for symbol in ordered]

    def expand_product(
        self, lhs: Sequence[Pattern], shapes: Sequence[Shape], verify: bool = False
    ) -> Solution | None:
        """
        Finds the family of identities

            product of phi_A = sum of c(B) * (product of phi_B)

        over every product B with the shapes, as a Solution, or None if there is
        none. With verify=True the family is also confirmed by verify_expansion.
        """
        lhs = tuple(lhs)
        for pattern in lhs:
            check_pattern(pattern, len(self.marked))
        _check_shapes([tuple(len(set(pattern)) for pattern in lhs), *shapes])
        products = self.products(shapes)
        logger.info(f'Solving for {len(products)} unknowns')
        if not products:
            return None

        equations = self._exact_equations(products, self._points(products), lhs)
        values = solve_modular(equations, len(products))
        if values is None:
            return None

        solution = Solution(lhs, dict(zip(products, values)))
        if verify:
            # Can only fail if the random points were degenerate
            solution.verified = self.verify_expansion(solution)
        return solution

    def verify_expansion(self, solution: Solution) -> bool:
        """
        Confirms exactly that every member of a family from expand_product is an
        identity on the graph. See docs/verification.md.
        """
        polynomials = lambda patterns: [
            self.graph.forest_polynomial(self.marked, p) for p in patterns
        ]
        products = list(solution.coefficients)
        logger.info(f'Verifying solution by expanding {len(products) + 1} products')
        equations = expansion_equations(
            polynomials(solution.lhs),
            [polynomials(product) for product in products],
            self.graph.num_edges,
        )

        coefficients = [_affine_parts(solution.coefficients[product]) for product in products]
        for row, constant in equations:
            totals = defaultdict(Fraction, {1: Fraction(-constant)})
            for entry, parts in zip(row, coefficients):
                if entry:
                    for symbol, value in parts.items():
                        totals[symbol] += entry * value
            if any(totals.values()):
                return False
        return True

    def _points(self, products: list[Product]) -> list[Point]:
        return random_points(len(products) + self.extra_points, self.graph.num_edges, self.seed)

    def _kernel(self, products: list[Product]) -> IntArray:
        """A basis modulo PRIME of the relations between the products, one row per relation."""
        if not products:
            return np.zeros((0, 0), dtype=np.int64)
        reduced, pivots = self._echelon_form(products)

        is_pivot = set(pivots)
        free = [column for column in range(len(products)) if column not in is_pivot]
        kernel = np.zeros((len(free), len(products)), dtype=np.int64)
        kernel[np.arange(len(free)), free] = 1
        kernel[:, list(pivots)] = -reduced[:, free].T % PRIME
        return kernel

    def _echelon_form(self, products: list[Product]) -> tuple[IntArray, tuple[int, ...]]:
        """The reduced rows and pivots modulo PRIME of the products at random points."""
        return reduced_row_echelon(self._product_matrix(products, self._points(products)), PRIME)

    def _inherited(self, factors: int, total: int, index: dict[Product, int]) -> IntArray:
        """
        The relations with one factor fewer times a forest polynomial, as one
        row per relation and polynomial in the coordinates of index.
        """
        num_marked = len(self.marked)
        relation_rows: list[IntArray] = [np.zeros((0, len(index)), dtype=np.int64)]
        for parts in _part_counts(num_marked):
            if factors == 1:
                break
            lower_products = self.products(all_shapes(num_marked, factors - 1, total - parts))
            kernel = self._kernel(lower_products)
            if len(kernel) == 0:
                continue
            for pattern in self.graph.forest_polynomials(self.marked, parts):
                multiples = np.zeros((len(kernel), len(index)), dtype=np.int64)
                columns = [index[_with_factor(product, pattern)] for product in lower_products]
                multiples[:, columns] = kernel
                relation_rows.append(multiples)
        return np.vstack(relation_rows)

    def _exact_equations(
        self, products: list[Product], points: list[Point], lhs: Product | None = None
    ) -> list[Equation]:
        """
        The (row, constant) equations, with exact values of the products at each
        point. The constant is the value of the given product, and 0 without one.
        """
        patterns = _patterns([*products, lhs]) if lhs else _patterns(products)
        values = {
            pattern: evaluate_all_numpy(self.graph.forest_polynomial(self.marked, pattern), points)
            for pattern in patterns
        }
        return [
            (
                tuple(
                    math.prod(values[pattern][point] for pattern in product)
                    for product in products
                ),
                math.prod(values[pattern][point] for pattern in lhs) if lhs else 0,
            )
            for point in range(len(points))
        ]

    def _product_matrix(self, products: list[Product], points: list[Point]) -> IntArray:
        """The (points, products) array of the values of the products modulo PRIME."""
        patterns = _patterns(products)
        index = {pattern: i for i, pattern in enumerate(patterns)}
        values = np.array(
            [
                [
                    value % PRIME
                    for value in evaluate_all_numpy(
                        self.graph.forest_polynomial(self.marked, pattern), points
                    )
                ]
                for pattern in patterns
            ],
            dtype=np.int64,
        )

        factors = np.array([[index[pattern] for pattern in product] for product in products])
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
    choices = itertools.combinations_with_replacement(parts, factors)
    return [shape for shape in choices if sum(shape) == total]


def format_relation(relation: Relation) -> str:
    """A relation from find_relations with one term per line, as printed by the command line."""
    lines = []
    for product, coefficient in relation.items():
        text = ' * '.join(f'phi_{pattern}' for pattern in product)
        lines.append(f'    {"+" if coefficient > 0 else "-"} {abs(coefficient)} * {text}')
    return '\n'.join(lines)


def _check_shapes(shapes: Iterable[Shape]) -> None:
    if len({(len(shape), sum(shape)) for shape in shapes}) > 1:
        raise ValueError('every shape needs the same number of factors and total number of parts')


def _part_counts(num_marked: int) -> range:
    return range(1, min(num_marked, MAX_PARTS) + 1)


def _with_factor(product: Product, pattern: Pattern) -> Product:
    """A product with one more factor, sorted as in Marking.products."""
    return tuple(sorted((*product, pattern), key=lambda factor: (len(set(factor)), factor)))


def _affine_parts(coefficient: sym.Expr) -> dict[Any, Fraction]:
    """An affine sympy expression as a dict from its symbols, and 1, to Fractions."""
    parts = cast(dict[Any, sym.Rational], coefficient.as_coefficients_dict())
    return {symbol: Fraction(int(value.p), int(value.q)) for symbol, value in parts.items()}


def _primitive(relation: dict[Product, Any]) -> Relation:
    """A relation rescaled to coprime integer coefficients, with the first positive."""
    denominator = math.lcm(*(coefficient.q for coefficient in relation.values()))
    integers = {
        product: int(coefficient * denominator) for product, coefficient in relation.items()
    }
    divisor = math.gcd(*integers.values())
    if next(iter(integers.values())) < 0:
        divisor = -divisor
    return {product: coefficient // divisor for product, coefficient in integers.items()}


def _patterns(products: Iterable[Product]) -> list[Pattern]:
    return sorted(set(itertools.chain.from_iterable(products)))
