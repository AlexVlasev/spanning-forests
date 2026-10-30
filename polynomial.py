from math import sqrt


class Polynomial:
    def __init__(self, variables, coefficients):
        self.variables = variables if variables else []
        self.coefficients = coefficients if coefficients else dict()

    def reciprocal(self):
        coefficients = dict()
        for power, coeff in self.coefficients.items():
            coefficients[tuple([-t for t in power])] = coeff
        return Polynomial(self.variables, coefficients)

    def __neg__(self):
        coefficients = {p: -c for p, c in self.coefficients.items()}
        return Polynomial(self.variables, coefficients)

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            coefficients = {p: c for p, c in self.coefficients.items()}
            zero = tuple((0 for _ in self.variables))
            if zero not in coefficients:
                coefficients[zero] = 0
            coefficients[zero] += other
            return Polynomial(self.variables, coefficients)
        if not isinstance(other, Polynomial):
            raise Exception(f'Not a polynomial: {type(other)}')
        if self.variables != other.variables:
            raise Exception('Not implemented yet')

        powers = set(self.coefficients.keys()).union(other.coefficients.keys())
        coefficients = dict()
        for p in powers:
            value = self.coefficients.get(p, 0) + other.coefficients.get(p, 0)
            if value == 0:
                continue
            coefficients[p] = value

        return Polynomial(self.variables, coefficients)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Polynomial(self.variables, {p: c * other for p, c in self.coefficients.items()})
        if not isinstance(other, Polynomial):
            raise Exception(f'Not a polynomial: {type(other)}')
        if self.variables != other.variables:
            raise Exception('Not implemented yet')

        new_coefficients = dict()
        for p, c in self.coefficients.items():
            for q, d in other.coefficients.items():
                r = tuple(i + j for i, j in zip(p, q))
                if r not in new_coefficients:
                    new_coefficients[r] = 0
                new_coefficients[r] += c * d

        return Polynomial(self.variables, new_coefficients)

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return not self.coefficients
        if not isinstance(other, Polynomial):
            raise Exception(f'Not a polynomial: {type(other)}')
        if self.variables != other.variables:
            raise Exception('Not implemented yet')

        ps = set(self.coefficients.keys())
        other_ps = set(other.coefficients.keys())
        if ps != other_ps:
            return False
        for p in ps:
            if self.coefficients[p] != other.coefficients[p]:
                return False
        return True

    def __repr__(self):
        # parts = [
        #     'Polynomial(',
        #     f'vars={self.variables}, ',
        #     f'coeffs={str(self.coefficients)}',
        #     ')',
        # ]
        # return ''.join(parts)
        return str(self)

    def __monomial(self, power):
        elements = []
        for p, x in zip(power, self.variables):
            if p == 0:
                continue
            if p == 1:
                elements.append(x)
                continue
            superscript = str(p)
            if superscript.startswith('-'):
                superscript = f'({superscript})'
            elements.append(f'{x}^{superscript}')

        return ''.join(sorted(elements))

    def __str__(self):
        terms = []
        items = sorted(self.coefficients.items(), key=lambda x: x[0])
        for p, c in items:
            if c == 0:
                continue
            monomial = self.__monomial(p)
            coefficient = c
            if isinstance(coefficient, Polynomial):
                term = f'({coefficient}){monomial}'
                terms.append(term)
                continue
            if c == 1:
                coefficient = ''
            if c == -1:
                coefficient = '-'
            term = f'{coefficient}{monomial}'
            if c < 0:
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

    def apply(self, variable, value):
        index = -1
        for i, var in enumerate(self.variables):
            if var == variable:
                index = i
        if index == -1:
            return self

        variables = self.variables.copy()
        variables.pop(index)

        coefficients = dict()
        for p, c in self.coefficients.items():
            power = list(p)
            power[index] = 0
            power = tuple(power)
            if power not in coefficients:
                coefficients[power] = 0
            coefficients[power] += c * value ** p[index]

        return Polynomial(self.variables, coefficients)

    def collect(self, vars_to_keep):
        new_vars = self.variables[:vars_to_keep]
        rest_vars = self.variables[vars_to_keep:]
        coefficients = dict()
        for power, coeff in self.coefficients.items():
            new_power = tuple(power[:vars_to_keep])
            rest_power = tuple(power[vars_to_keep:])
            if new_power not in coefficients:
                coefficients[new_power] = Polynomial(rest_vars, dict())
            poly = Polynomial(rest_vars, {rest_power: coeff})
            coefficients[new_power] = coefficients[new_power] + poly
        return Polynomial(new_vars, coefficients)

    def truncate(self, variable, power):
        index = -1
        for i, var in enumerate(self.variables):
            if variable == var:
                index = i
                break
        if index < 0:
            raise Exception(f'Variable "{variable}" is not one of the variables of the polynomial')

        coefficients = dict()
        for p, value in self.coefficients.items():
            if p[index] >= power:
                continue
            coefficients[p] = value

        return Polynomial(self.variables, coefficients)


if __name__ == '__main__':
    # variables = ['x', 'y']
    # P = Polynomial(variables, {(1, 0): 2, (0, 1): 3, (1, 1): 1})
    # # Q = Polynomial(variables, {(0, 0): 1, (0, 1): 2})

    # print(P)

    # Pc = P.collect(1)
    # print(Pc)

    # N = 7
    # variables = ['u']
    # u = Polynomial(variables, {(1,0): 1})
    # y = u
    # for i in range(N):
    #     y = (u + y*y).truncate('u', i+1)
    #     print(y)

    N = 16
    variables = ['x']
    u = Polynomial(variables, {(2, 0): 0.5})
    y = u
    for i in range(N):
        y = (u + y * y).truncate('x', 2 * i)
        # print(y)

    coefficients = dict()
    for p, value in y.coefficients.items():
        coefficients[p] = value / (p[0] + 1)
    integral = Polynomial(variables, coefficients) * Polynomial(variables, {(1,): 1})
    print(integral.apply('x', 1))

    variables = ['x']
    u = Polynomial(variables, {(2, 0): 1})
    y = u
    for i in range(N):
        y = (u + y * y).truncate('x', 2 * i)
        # print(y)

    coefficients = dict()
    for p, value in y.coefficients.items():
        coefficients[p] = value / (p[0] + 1)
    integral = Polynomial(variables, coefficients) * Polynomial(variables, {(1,): 1})
    print(integral.apply('x', 1 / sqrt(2)) * sqrt(2))
