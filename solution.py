"""
Families of quadratic identities between spanning forest polynomials, as
found by IdentitySolver.
"""

import sympy as sym


class Solution:
    """
    A family of identities

        phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

    with the coefficients c keyed by (P_i, Q_j) pattern pairs. Coefficients
    are sympy expressions, affine in the free parameters of the family.
    verified is True or False once the family has been checked
    symbolically, and None before.
    """

    def __init__(self, P, Q, coefficients, verified=None):
        self.P = P
        self.Q = Q
        self.coefficients = {pair: sym.sympify(c) for pair, c in coefficients.items()}
        self.verified = verified

    @property
    def free_parameters(self):
        symbols = set().union(*(c.free_symbols for c in self.coefficients.values()))
        return sorted(symbols, key=str)

    @property
    def dimension(self):
        return len(self.free_parameters)

    def classify(self):
        return classify(self.coefficients)

    def report(self):
        """A summary of the coefficients of the family, as printed by the command line."""
        classes = self.classify()
        lines = []
        for title, kind in (('Depd', 'dependent'), ('Fixd', 'fixed'), ('Free', 'free')):
            for (p, q), value in classes[kind]:
                lines.append(f'{title}: {p} x {q} = {value}')

        lines.append('\n====================================\n')
        lines.append(f'Num vars : {len(self.coefficients)}')
        lines.append(f'Dependent: {len(classes["dependent"])}')
        lines.append(f'Fixed    : {len(classes["fixed"])}')
        lines.append(f'Dimension: {len(classes["free"])}')
        lines.append(f'Forbidden: {len(classes["forbidden"])}')
        return '\n'.join(lines)

    def identity(self):
        """The identity, listing the nonzero terms of the right hand side."""
        lines = [f'phi_{self.P} * phi_{self.Q} =']
        for (p, q), c in self.coefficients.items():
            if c != 0:
                lines.append(f'    + ({c}) * phi_{p} * phi_{q}')
        return '\n'.join(lines)


def classify(values):
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
