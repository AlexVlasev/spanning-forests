import math
import itertools
import numpy as np
import sympy as sym

from time import time
from scipy.linalg import lu

from polynomial import Polynomial
from matrix import Matrix

DEBUG = True

"""
1. Define a graph, whether random or like the complete graph K_n
2. Build out the edge laplacian for the complementa of spanning trees
3. Compute the psi_G polynomial and invert to capture spanning trees instead
4. Convert psi_G monomials into spanning trees
5. For each spanning tree, find the resulting m-forest collections
6. Combine all the collections from all spanning trees
7. For each forest in the collection, compute the resulting partition and forest monomial
8. Collect monomials by partition and add them to form the phi_G polynomials
"""

# 1. Define a graph, whether random or like K_n

K3 = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]


def get_kn(n):
    kn = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    return kn


# template = [
#     [0, , , , ],
#     [0, 0, , , ],
#     [0, 0, 0, , ],
#     [0, 0, 0, 0, ],
#     [0, 0, 0, 0, 0],

HOUSE = [
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
]

HOUSE = Matrix(HOUSE)
HOUSE += HOUSE.transpose()
HOUSE = HOUSE.array

MATRIX = get_kn(5)

VER_5 = [0, 1, 2]
VERT = [0, 1, 2, 3]

VERTICES = VERT

num_vertices = len(MATRIX)
# print(MATRIX)
num_edges = int(sum((sum(row) for row in MATRIX)) / 2)

VAR_EMPTY = []
VAR_ABC = ['b', 'a', 'g', 'e', 'd', 'f', 'c']
alphabet = 'abcdefghijklmnoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
VAR_ABC_N = [alphabet[i] for i in range(num_edges)]

REPLACE_VARS = VAR_ABC_N


class Solver:
    def __init__(self, matrix, display_vars=None):
        self.matrix = matrix
        self.display_vars = display_vars
        self.num_vertices = len(self.matrix)
        self.num_edges = int(sum((sum(row) for row in self.matrix)) / 2)

        self.edges = []
        self.edge_vars = []

        self.mat = None
        self.graph_poly = None
        self.tree_poly = None

        self.vertices = None

        self._initialize()
        self.debug_print = DEBUG

    def _initialize(self):
        """
        Computes the graph polynomial and its reciprocal tree polynomial
        by first computing an incidence matrix and then using the matrix
        tree theorem.
        """
        incidence = [[0 for i in range(self.num_edges)] for j in range(self.num_vertices)]

        num_edges = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.matrix[i][j] == 0:
                    continue

                self.edges.append((i, j))
                self.edge_vars.append(f'e({i},{j})')

                incidence[i][num_edges] = 1
                incidence[j][num_edges] = -1
                num_edges += 1

        if self.display_vars is None:
            self.display_vars = self.edge_vars

        incidence.pop(-1)
        incidence_T = Matrix(incidence).transpose().array

        # Getting the Laplacian incidence matrix
        diag = [[0 for i in range(num_edges)] for j in range(num_edges)]
        for i, edge in enumerate(self.edge_vars):
            power = tuple((1 if j == i else 0 for j in range(num_edges)))
            diag[i][i] = Polynomial(self.edge_vars, {power: 1})
        self.mat = Matrix.expand_block(diag, incidence_T, (-Matrix(incidence)).array)

        # Matrix tree theorem
        self.graph_poly = self.mat.det()
        self.graph_poly = Polynomial(self.edge_vars, self.graph_poly.coefficients)

        # The reciprocal polynomial which has all the edges from the trees (not the complement)
        all_vars = tuple([1 for _ in range(self.num_edges)])
        nomial = Polynomial(self.edge_vars, {all_vars: 1})
        self.tree_poly = nomial * self.graph_poly.reciprocal()

    def set_vertices(self, vertices):
        self.vertices = vertices

    def _get_trees(self):
        """Each spanning tree as the set of indices of its edges."""
        return [
            frozenset(i for i, p in enumerate(power) if p) for power in self.tree_poly.coefficients
        ]

    def _get_pattern(self, forest):
        """
        Labels each marked vertex by the tree of the forest containing it,
        numbering the trees in order of first appearance, e.g. '1213'.
        """
        parent = list(range(self.num_vertices))

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        for e in forest:
            i, j = self.edges[e]
            parent[find(i)] = find(j)

        labels = dict()
        return ''.join(str(labels.setdefault(find(v), len(labels) + 1)) for v in self.vertices)

    def get_sfp_for(self, num_partitions):
        """
        Computes the spanning forest polynomial of every partition of the
        marked vertices into num_partitions parts, keyed by pattern.

        Every spanning forest with k trees is a spanning tree minus k - 1 of
        its edges, so removing edges from the trees finds them all. A forest
        whose marked vertices span fewer than k trees has a tree without any
        marked vertex, so it belongs to no pattern.
        """
        forests = set()
        for tree in self._get_trees():
            for removed in itertools.combinations(tree, num_partitions - 1):
                forests.add(tree.difference(removed))

        coefficients = dict()
        for forest in forests:
            pattern = self._get_pattern(forest)
            if len(set(pattern)) != num_partitions:
                continue
            power = tuple(1 if e in forest else 0 for e in range(self.num_edges))
            coefficients.setdefault(pattern, dict())[power] = 1

        return {p: Polynomial(self.edge_vars, coefficients[p]) for p in sorted(coefficients)}

    def get_phi_P(self, P):
        assert len(P) == len(self.vertices)
        if set(P) == {'1'}:
            return self.tree_poly

        length = len(set(P))
        sfps = self.get_sfp_for(length)
        return sfps.get(P, 0)

    def print_poly(self, poly: Polynomial):
        poly.variables = self.display_vars
        print(f'Poly: {poly}')
        poly.variables = self.edge_vars

    def _convert_to_vector_form(self, variables, polynomials):
        matrix = []
        vector = []
        # n = max(len(polynomials), len(variables))
        n = len(variables)
        for poly in polynomials:
            matrix_row = [0 for _ in range(n)]
            vector_elem = 0
            print(poly)
            for p, c in poly.coefficients.items():
                if sum(p) == 0:
                    vector_elem = c
                    continue
                for index, entry in enumerate(p):
                    if entry != 0:
                        matrix_row[index] = c
            print(matrix_row)
            print(vector_elem)
            matrix.append(matrix_row)
            vector.append(vector_elem)
        return matrix, vector

    def solve_matrix(self, matrix, vector):
        m = np.array(matrix)
        v = np.array(vector)
        U = lu(m)[2]
        basis_columns = {np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])}
        free_variables = set(range(U.shape[1])) - basis_columns
        self.debug(f'Free variables in LU: {len(free_variables)}')
        self.debug(f'Free variables in LU: {free_variables}')
        # x = np.linalg.solve(m, v)
        # print(x)

    def get_lhs(self, P, Q):
        self.debug(f'Getting {P} polynomial')
        phi_P = self.get_phi_P(P)
        self.debug(f'Polynomial {P} has {len(phi_P.coefficients.items())} terms')
        # self.print_poly(phi_P)

        self.debug(f'Getting {Q} polynomial')
        phi_Q = self.get_phi_P(Q)
        self.debug(f'Polynomial {Q} has {len(phi_Q.coefficients.items())} terms')
        # self.print_poly(phi_Q)

        return phi_P, phi_Q

    def get_rhs(self, P, Q, reduce=False):
        p, q = len(set(P)), len(set(Q))
        a, b = p + 1, q - 1
        if reduce:
            mod = (p + q) % 2
            a = int((p + q - mod) / 2)
            b = int((p + q + mod) / 2)

        self.debug(f'Getting P_k polynomials. Partitions: {a}')
        sfps_P = list(self.get_sfp_for(a).items())
        self.debug(f'Found {len(sfps_P)} P_k polynomials')

        self.debug(f'Getting Q_k polynomials. Partitions: {b}')
        sfps_Q = list(self.get_sfp_for(b).items())
        self.debug(f'Found {len(sfps_Q)} P_k polynomials')

        return sfps_P, sfps_Q

    def get_abstract_vars(self, sfps_P, sfps_Q):
        num_P = len(sfps_P)
        num_Q = len(sfps_Q)

        variables = []
        reverse_variables = dict()
        var_map = dict()

        index = 0
        for i in range(num_P):
            p = sfps_P[i][0]
            for j in range(num_Q):
                q = sfps_Q[j][0]
                variables.append(f'a({i},{j})')
                reverse_variables[(i, j)] = index
                var_map[(p, q)] = (i, j)
                index += 1

        return variables, reverse_variables, var_map

    def expand_polynomials(self, phi_P, phi_Q, sfps_P, sfps_Q, variables, zeros):
        self.debug(f'Expanding polynomial vars for identity')
        phi_P = self.expand_poly(phi_P, variables, zeros)
        phi_Q = self.expand_poly(phi_Q, variables, zeros)

        self.debug('Partition patterns')

        sfps_P = [(p, self.expand_poly(poly, variables, zeros)) for p, poly in sfps_P]
        self.debug('\nP_k:\n')
        for p, _ in sfps_P:
            self.debug(p)

        sfps_Q = [(p, self.expand_poly(poly, variables, zeros)) for p, poly in sfps_Q]
        self.debug('\nQ_k:\n')
        for p, _ in sfps_Q:
            self.debug(p)

        return phi_P, phi_Q, sfps_P, sfps_Q

    def get_polynomial(self, phi_P, phi_Q, sfps_P, sfps_Q, variables, reverse_variables, var_map):
        self.debug(f'\nMultiplying P, Q polynomial pairs: {len(sfps_P) * len(sfps_Q) + 1}')
        # Encodes phi_P * phi_Q = sum of a(i,j) * phi_P_k * phi_Q_k as
        # -phi_P * phi_Q + sum of a(i,j) * phi_P_k * phi_Q_k = 0
        polynomials = [-(phi_P * phi_Q)]

        for p, p_poly in sfps_P:
            for q, q_poly in sfps_Q:
                var_coords = var_map[(p, q)]
                index = reverse_variables[var_coords]

                power = [0 for i in range(len(variables))]
                power[index + self.num_edges] = 1
                coeffs = {tuple(power): 1}

                var_poly = Polynomial(variables, coeffs)
                # print(p, q, var_coords, index, power, var_poly)

                prod = p_poly * q_poly * var_poly
                # print(prod)
                polynomials.append(prod)

        return sum(polynomials)

    def simplify_polynomial(self, polynomial_to_simplify):
        polynomial = polynomial_to_simplify.collect(self.num_edges)
        self.debug(f'Simplified polynomial has {len(polynomial.coefficients.items())} terms')
        # for power, coeff in simplified.coefficients.items():
        #     print(f'{power}: {coeff}')

        zero_vars = set()
        for p, c in polynomial.coefficients.items():
            if not isinstance(c, Polynomial):
                continue
            # A monomial whose coefficient is a single a(i,j) forces it to be 0
            if len(c.coefficients) == 1:
                (power,) = c.coefficients
                if any(power):
                    zero_vars.add(c.variables[power.index(1)])

        self.debug(f'Number of zero variables: {len(zero_vars)}')

        self.debug(f'Simplifying polynomial further')
        for var in zero_vars:
            new_coefficients = {p: c.apply(var, 0) for p, c in polynomial.coefficients.items()}
            polynomial = Polynomial(self.edge_vars, new_coefficients)

            new_coefficients = dict()
            for p, poly in polynomial.coefficients.items():
                gs = set()
                for pp, c in poly.coefficients.items():
                    gs.add(c)
                gcd = math.gcd(*gs)
                new_poly = poly
                if gcd > 1:
                    new_poly_coeffs = {pp: int(c / gcd) for pp, c in poly.coefficients.items()}
                    new_poly = Polynomial(poly.variables, new_poly_coeffs)
                    # print(f'GCD found: {gcd}: {poly} -> {new_poly}')
                new_coefficients[p] = new_poly

            polynomial = Polynomial(self.edge_vars, new_coefficients)

        return polynomial, zero_vars

    def get_equations(self, polynomial):
        self.debug(
            f'Removing duplicate equations from {len(polynomial.coefficients.items())} equations'
        )
        equations = []

        # terms = []
        for p, c in polynomial.coefficients.items():
            if c in equations:
                continue
            equations.append(c)
            # print(c)
            # terms.extend(str(c).split(' + '))

        # terms = set(terms)
        # self.debug(f'Found {len(terms)} terms: {terms}')

        # for eq in equations:
        #     print(f'{eq} = 0')

        self.debug(f'Remaining equations: {len(equations)}')

        return equations

    def solve_system(self, equations, a_vars):
        # self.debug('Computing LU decomposition')
        # matrix, vector = self._convert_to_vector_form(a_vars, equations)
        # print(matrix)
        # print(vector)
        # self.solve_matrix(matrix, vector)

        self.debug(f'Extracting linear system')
        xs = sym.symbols(f'x_:{len(a_vars)}')
        rows = []
        for poly in equations:
            p = poly
            for x, v in zip(xs, a_vars):
                p = p.apply(v, x)
            # Substituting every a(i,j) moves all terms onto the constant
            # monomial, so the equation is the sum of the coefficients = 0
            rows.append(sum(p.coefficients.values()))

        self.debug(f'Solving linear system')
        # soln = sym.nonlinsolve(rows, xs)
        soln = sym.linsolve(rows, xs)

        return soln

    def solve_problem(self, P, Q, reduce=False):
        phi_P, phi_Q = self.get_lhs(P, Q)
        sfps_P, sfps_Q = self.get_rhs(P, Q, reduce=reduce)
        variables, reverse_variables, var_map = self.get_abstract_vars(sfps_P, sfps_Q)

        a_vars = variables
        self.debug(f'We have {len(a_vars)} a_i,j variables')

        variables = [*self.edge_vars, *variables]
        zeros = [0 for _ in range(len(a_vars))]
        phi_P, phi_Q, sfps_P, sfps_Q = self.expand_polynomials(
            phi_P, phi_Q, sfps_P, sfps_Q, variables, zeros
        )

        polynomial = self.get_polynomial(
            phi_P, phi_Q, sfps_P, sfps_Q, variables, reverse_variables, var_map
        )
        polynomial, zero_vars = self.simplify_polynomial(polynomial)

        equations = self.get_equations(polynomial)
        soln = self.solve_system(equations, a_vars)

        # linsolve returns either EmptySet or a FiniteSet with one parametric solution
        if not soln.args:
            print('\n====================================\n')
            print('No Solution')
            return None
        values = {
            var: 0 if var in zero_vars else value for var, value in zip(a_vars, soln.args[0].args)
        }

        classes = self.classify_solution(values)
        for var, value in classes['dependent']:
            print(f'Depd: {var} = {value}')
        for var, value in classes['fixed']:
            print(f'Fixd: {var} = {value}')
        for var, value in classes['free']:
            print(f'Free: {var} = {value}')

        print('\n====================================\n')
        print(f'Num vars : {len(a_vars)}')
        print(f'Dependent: {len(classes["dependent"])}')
        print(f'Fixed    : {len(classes["fixed"])}')
        print(f'Dimension: {len(classes["free"])}')
        print(f'Forbidden: {len(classes["forbidden"])}')

        print('\n====================================\n')
        print(tuple(values.values()))

        # Map each (P_k, Q_k) pattern pair to its coefficient a(i,j) in
        # phi_P * phi_Q = sum of a(i,j) * phi_P_k * phi_Q_k
        return {pair: values[a_vars[reverse_variables[coords]]] for pair, coords in var_map.items()}

    @staticmethod
    def classify_solution(values):
        """
        Sorts the coefficients of a solution into exactly one of:
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

    def expand_poly(self, poly, variables, zeros):
        new_coeffs = dict()
        for power, coeff in poly.coefficients.items():
            new_power = [*power, *zeros]
            new_coeffs[tuple(new_power)] = coeff
        return Polynomial(variables, new_coeffs)

    def debug(self, expr):
        if self.debug_print:
            print(expr)


if __name__ == '__main__':
    start = time()

    # n = 5
    # vertices = [k for k in range(n-1)]
    # solver = Solver(get_kn(n), display_vars=None)
    # solver.set_vertices(vertices)
    # P = '1' * (n-1)
    # Q = ''.join(map(lambda x: str(x+1), vertices))
    # solver.solve_problem(P, Q)

    # vertices = [0, 1, 2, 3, 4]
    # solver = Solver(get_kn(6), display_vars=None)
    # solver.set_vertices(vertices)
    # P = '11111'
    # Q = '12345'
    # solver.solve_problem(P, Q, reduce=True)

    # vertices = [0, 1, 2]
    # solver = Solver(get_kn(5), display_vars=None)
    # solver.set_vertices(vertices)
    # P = '111'
    # Q = '123'
    # solver.solve_problem(P, Q, reduce=True)

    # vertices = [0, 1, 2, 3, 4]
    # solver = Solver(get_kn(6), display_vars=None)
    # solver.set_vertices(vertices)
    # P = '11122'
    # Q = '12345'
    # solver.solve_problem(P, Q, reduce=False)

    vertices = [0, 1, 2, 3]
    solver = Solver(get_kn(5), display_vars=None)
    solver.set_vertices(vertices)
    P = '1111'
    Q = '1234'
    solver.solve_problem(P, Q, reduce=True)

    end = time()
    print(f'\nComputation took: {int(100*(end-start))/100} sec')
