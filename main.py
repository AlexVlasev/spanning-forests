import math
import itertools
import sympy as sym

from time import time

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

        self.trees = []
        self.graph_poly = None
        self.tree_poly = None

        self.vertices = None

        self._initialize()
        self.debug_print = DEBUG

    def _initialize(self):
        """
        Enumerates the spanning trees and builds from them the tree polynomial
        (edges in each tree) and the graph polynomial (edges not in each tree).
        """
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.matrix[i][j] == 0:
                    continue
                self.edges.append((i, j))
                self.edge_vars.append(f'e({i},{j})')

        if self.display_vars is None:
            self.display_vars = self.edge_vars

        self.trees = self._enumerate_trees()

        tree_coefficients = dict()
        graph_coefficients = dict()
        for tree in self.trees:
            tree_coefficients[tuple(1 if e in tree else 0 for e in range(self.num_edges))] = 1
            graph_coefficients[tuple(0 if e in tree else 1 for e in range(self.num_edges))] = 1
        self.tree_poly = Polynomial(self.edge_vars, tree_coefficients)
        self.graph_poly = Polynomial(self.edge_vars, graph_coefficients)

    def _is_connected(self, excluded):
        """Whether the edges not in excluded connect every vertex."""
        neighbours = [[] for _ in range(self.num_vertices)]
        for e, (i, j) in enumerate(self.edges):
            if e not in excluded:
                neighbours[i].append(j)
                neighbours[j].append(i)

        seen = {0}
        stack = [0]
        while stack:
            for w in neighbours[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == self.num_vertices

    def _enumerate_trees(self):
        """
        Lists every spanning tree as the set of indices of its edges.

        Walks the edges in order, deciding for each whether the tree contains
        it. An edge is only included if it joins two different trees of the
        forest chosen so far, and only excluded if the remaining edges still
        connect the graph, so every branch of the search ends in a spanning
        tree.
        """
        trees = []
        excluded = set()

        def search(e, chosen, components):
            if len(chosen) == self.num_vertices - 1:
                trees.append(frozenset(chosen))
                return

            i, j = self.edges[e]
            ci, cj = components[i], components[j]
            if ci != cj:
                merged = [ci if c == cj else c for c in components]
                search(e + 1, [*chosen, e], merged)

            excluded.add(e)
            if self._is_connected(excluded):
                search(e + 1, chosen, components)
            excluded.remove(e)

        if self._is_connected(excluded):
            search(0, [], list(range(self.num_vertices)))
        return trees

    def set_vertices(self, vertices):
        self.vertices = vertices

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
        for tree in self.trees:
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

    def get_equations(self, phi_P, phi_Q, sfps_P, sfps_Q):
        """
        Builds the linear system for the a(i,j) in

            phi_P * phi_Q = sum of a(i,j) * phi_P_i * phi_Q_j

        with one equation per edge monomial, saying that its coefficients on
        both sides agree. Each equation is returned as a pair (row, constant)
        where row holds the coefficient of each a(i,j), in the order of
        get_abstract_vars, reduced by their gcd. Duplicates are dropped.
        """
        self.debug(f'\nMultiplying P, Q polynomial pairs: {len(sfps_P) * len(sfps_Q) + 1}')
        lhs = (phi_P * phi_Q).coefficients
        products = [(p_poly * q_poly).coefficients for _, p_poly in sfps_P for _, q_poly in sfps_Q]

        monomials = set(lhs).union(*products)
        self.debug(f'Removing duplicate equations from {len(monomials)} equations')
        equations = set()
        for monomial in monomials:
            row = tuple(product.get(monomial, 0) for product in products)
            constant = lhs.get(monomial, 0)
            gcd = math.gcd(*row, constant)
            if gcd == 0:
                continue
            equations.add((tuple(c // gcd for c in row), constant // gcd))

        self.debug(f'Remaining equations: {len(equations)}')
        return equations

    def solve_system(self, equations, a_vars):
        self.debug(f'Solving linear system')
        xs = sym.symbols(f'x_:{len(a_vars)}')
        # Sort to keep the parametrization of the solution reproducible
        equations = sorted(equations)
        A = sym.Matrix([row for row, _ in equations])
        b = sym.Matrix([constant for _, constant in equations])
        return sym.linsolve((A, b), xs)

    def solve_problem(self, P, Q, reduce=False):
        phi_P, phi_Q = self.get_lhs(P, Q)
        sfps_P, sfps_Q = self.get_rhs(P, Q, reduce=reduce)
        a_vars, reverse_variables, var_map = self.get_abstract_vars(sfps_P, sfps_Q)
        self.debug(f'We have {len(a_vars)} a_i,j variables')

        equations = self.get_equations(phi_P, phi_Q, sfps_P, sfps_Q)
        soln = self.solve_system(equations, a_vars)

        # linsolve returns either EmptySet or a FiniteSet with one parametric solution
        if not soln.args:
            print('\n====================================\n')
            print('No Solution')
            return None
        values = dict(zip(a_vars, soln.args[0].args))

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
