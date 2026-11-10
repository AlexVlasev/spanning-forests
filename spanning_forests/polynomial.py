type Power = tuple[int, ...]


class Polynomial:
    def __init__(self, variables: list[str] | None, coefficients: dict[Power, int] | None) -> None:
        self.variables = variables if variables else []
        self.coefficients = coefficients if coefficients else dict()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.variables == other.variables and self._nonzero() == other._nonzero()

    def _nonzero(self) -> dict[Power, int]:
        """The coefficients that are not 0."""
        return {
            power: coefficient
            for power, coefficient in self.coefficients.items()
            if coefficient != 0
        }

    def __repr__(self) -> str:
        return str(self)

    def __monomial(self, power: Power) -> str:
        elements = []
        for exponent, variable in zip(power, self.variables):
            if exponent == 0:
                continue
            if exponent == 1:
                elements.append(variable)
                continue
            superscript = str(exponent)
            if superscript.startswith('-'):
                superscript = f'({superscript})'
            elements.append(f'{variable}^{superscript}')

        return ''.join(sorted(elements))

    def __str__(self) -> str:
        terms = []
        items = sorted(self.coefficients.items(), key=lambda item: item[0])
        for power, value in items:
            if value == 0:
                continue
            monomial = self.__monomial(power)
            coefficient = value
            if value == 1:
                coefficient = ''
            if value == -1:
                coefficient = '-'
            term = f'{coefficient}{monomial}'
            if value < 0:
                term = f'({term})'
            if not term:
                term = '1'
            if term == '(-)':
                term = '-1'
            terms.append(term)
        expr = ' + '.join(terms)
        if not expr:
            return '0'
        return expr


class MultilinearPolynomial:
    """
    A polynomial in which no variable appears squared, such as a spanning
    forest polynomial, with terms mapping the bitmask of the variables in
    each monomial (bit e for variable e) to its coefficient.
    """

    def __init__(self, variables: list[str], terms: dict[int, int]) -> None:
        self.variables = variables
        self.terms = dict(terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultilinearPolynomial):
            return NotImplemented
        return self.variables == other.variables and self.terms == other.terms

    def __mul__(self, scalar: int) -> 'MultilinearPolynomial':
        return MultilinearPolynomial(
            self.variables,
            {mask: coefficient * scalar for mask, coefficient in self.terms.items()},
        )

    __rmul__ = __mul__

    def to_polynomial(self) -> Polynomial:
        """The same polynomial as a Polynomial, with exponent tuples."""
        coefficients = dict()
        for mask, coefficient in self.terms.items():
            power = tuple((mask >> e) & 1 for e in range(len(self.variables)))
            coefficients[power] = coefficient
        return Polynomial(self.variables, coefficients)

    def __repr__(self) -> str:
        return str(self.to_polynomial())


def edge_indices(mask: int) -> tuple[int, ...]:
    """The indices of the set bits of a bitmask, from the lowest."""
    indices = []
    while mask:
        low = mask & -mask
        indices.append(low.bit_length() - 1)
        mask ^= low
    return tuple(indices)
