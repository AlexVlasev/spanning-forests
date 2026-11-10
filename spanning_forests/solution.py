"""
Families of identities between products of spanning forest polynomials, as
found by IdentitySolver and Marking.expand_product.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import sympy as sym

from spanning_forests.aliases import Pattern, Term


class Solution:
    """
    A family of identities

        phi_A_1 * ... * phi_A_r = sum of c(B_1, ..., B_s) * phi_B_1 * ... * phi_B_s

    with product the tuple of patterns A_i, and the coefficients c keyed by
    tuples of patterns B_j. Coefficients are sympy expressions, affine in
    the free parameters of the family.
    verified is True or False once the family has been checked
    symbolically, and None before.
    """

    def __init__(
        self,
        product: Sequence[Pattern],
        coefficients: Mapping[Any, Any],
        verified: bool | None = None,
    ) -> None:
        self.product = tuple(product)
        self.coefficients = {pair: sym.sympify(c) for pair, c in coefficients.items()}
        self.verified = verified

    @property
    def free_parameters(self) -> list[sym.Symbol]:
        symbols = set().union(*(c.free_symbols for c in self.coefficients.values()))
        return sorted(symbols, key=str)

    @property
    def dimension(self) -> int:
        return len(self.free_parameters)

    def classify(self) -> dict[str, list[tuple[Term, sym.Expr]]]:
        return classify(self.coefficients)

    def report(self) -> str:
        """A summary of the coefficients of the family, as printed by the command line."""
        classes = self.classify()
        lines = []
        for title, kind in (('Depd', 'dependent'), ('Fixd', 'fixed'), ('Free', 'free')):
            for term, value in classes[kind]:
                lines.append(f'{title}: {" x ".join(term)} = {value}')

        lines.append('\n====================================\n')
        lines.append(f'Num vars : {len(self.coefficients)}')
        lines.append(f'Dependent: {len(classes["dependent"])}')
        lines.append(f'Fixed    : {len(classes["fixed"])}')
        lines.append(f'Dimension: {len(classes["free"])}')
        lines.append(f'Forbidden: {len(classes["forbidden"])}')
        return '\n'.join(lines)

    def identity(self) -> str:
        """The identity, listing the nonzero terms of the right hand side."""
        lines = [f'{_format_product(self.product)} =']
        for term, c in self.coefficients.items():
            if c != 0:
                lines.append(f'    + ({c}) * {_format_product(term)}')
        return '\n'.join(lines)


def _format_product(patterns: Iterable[Pattern]) -> str:
    return ' * '.join(f'phi_{pattern}' for pattern in patterns)


def classify(values: Mapping[Any, Any]) -> dict[str, list[tuple[Term, sym.Expr]]]:
    """
    Sorts coefficients into exactly one of:
        forbidden: always 0
        free:      a parameter of the solution family
        fixed:     a nonzero constant
        dependent: determined by the free parameters
    """
    classes = {'forbidden': [], 'free': [], 'fixed': [], 'dependent': []}
    for key, value in values.items():
        value = sym.sympify(value)
        if value == 0:
            kind = 'forbidden'
        elif value.is_Symbol:
            kind = 'free'
        elif not value.free_symbols:
            kind = 'fixed'
        else:
            kind = 'dependent'
        classes[kind].append((key, value))
    return classes
