import math
import itertools
import random
import sympy as sym

from time import time

from polynomial import Polynomial

DEBUG = True

"""
1. Define a graph, whether random or like the complete graph K_n
2. Enumerate the spanning trees, giving the tree polynomial and its reciprocal psi_G
3. For each spanning tree, find the resulting m-forest collections
4. Combine all the collections from all spanning trees
5. For each forest in the collection, compute the resulting partition and forest monomial
6. Collect monomials by partition and add them to form the phi_G polynomials
"""

def get_kn(n):
    kn = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    return kn


class Solver:
    def __init__(self, matrix):
        self.matrix = matrix
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

    def get_lhs(self, P, Q):
        self.debug(f'Getting {P} polynomial')
        phi_P = self.get_phi_P(P)
        self.debug(f'Polynomial {P} has {len(phi_P.coefficients.items())} terms')

        self.debug(f'Getting {Q} polynomial')
        phi_Q = self.get_phi_P(Q)
        self.debug(f'Polynomial {Q} has {len(phi_Q.coefficients.items())} terms')

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

    def get_equations(self, phi_P, phi_Q, sfps_P, sfps_Q, chunk_edges=6):
        """
        Builds the linear system for the a(i,j) in

            phi_P * phi_Q = sum of a(i,j) * phi_P_i * phi_Q_j

        with one equation per edge monomial, saying that its coefficients on
        both sides agree. Each equation is returned as a pair (row, constant)
        where row holds the coefficient of each a(i,j), in the order of
        get_abstract_vars, reduced by their gcd. Duplicates are dropped.

        To limit memory, monomials are processed in chunks according to which
        of the first chunk_edges edges they contain, and which of those are
        squared. The product monomial (a | b, a & b) lands in the chunk
        (x | y, x & y), where x and y are the parts of a and b on those edges,
        so sorting the terms of every polynomial by that part lets each chunk
        multiply only its own terms. Every pair of terms is still multiplied
        exactly once.
        """
        self.debug(f'\nMultiplying P, Q polynomial pairs: {len(sfps_P) * len(sfps_Q) + 1}')
        chunk_mask = (1 << min(chunk_edges, self.num_edges)) - 1

        # Factor 0 is the left hand side, and factor k + 1 multiplies a(i,j) number k
        masks_P = [self._masks(poly) for _, poly in sfps_P]
        masks_Q = [self._masks(poly) for _, poly in sfps_Q]
        factors = [(self._masks(phi_P), self._masks(phi_Q))]
        factors.extend(itertools.product(masks_P, masks_Q))
        buckets = [(self._bucket(a, chunk_mask), self._bucket(b, chunk_mask)) for a, b in factors]

        chunks = dict()
        for x in range(chunk_mask + 1):
            for y in range(chunk_mask + 1):
                chunks.setdefault((x | y, x & y), []).append((x, y))

        self.debug(f'Building equations in {len(chunks)} chunks')
        sparse_equations = set()
        for pairs in chunks.values():
            sparse_equations.update(self._chunk_equations(buckets, pairs))

        num_vars = len(factors) - 1
        equations = set()
        for entries, constant in sparse_equations:
            row = [0] * num_vars
            for k, c in entries:
                row[k] = c
            equations.add((tuple(row), constant))

        self.debug(f'Remaining equations: {len(equations)}')
        return equations

    def _chunk_equations(self, buckets, pairs):
        """
        The equations of get_equations for the monomials in one chunk, with
        sparse rows of (index of a(i,j), coefficient) pairs.
        """
        products = []
        for buckets_a, buckets_b in buckets:
            product = dict()
            for x, y in pairs:
                self._multiply_into(product, buckets_a.get(x, ()), buckets_b.get(y, ()))
            products.append(product)

        lhs = products[0]
        rows = {monomial: [] for monomial in lhs}
        for k, product in enumerate(products[1:]):
            for monomial, c in product.items():
                rows.setdefault(monomial, []).append((k, c))

        equations = set()
        for monomial, entries in rows.items():
            constant = lhs.get(monomial, 0)
            gcd = math.gcd(*(c for _, c in entries), constant)
            if gcd == 0:
                continue
            equations.add((tuple((k, c // gcd) for k, c in entries), constant // gcd))
        return equations

    @staticmethod
    def _bucket(terms, mask):
        """Sorts terms from _masks by which of the edges in mask they contain."""
        buckets = dict()
        for term in terms:
            buckets.setdefault(term[0] & mask, []).append(term)
        return buckets

    @staticmethod
    def _masks(poly):
        """
        The terms of a polynomial without squares, such as a spanning forest
        polynomial, as (bitmask of edges, coefficient) pairs.
        """
        if not isinstance(poly, Polynomial):
            return []
        terms = []
        for power, c in poly.coefficients.items():
            assert all(k <= 1 for k in power), 'Expected a polynomial without squares'
            terms.append((sum(1 << e for e, k in enumerate(power) if k), c))
        return terms

    @staticmethod
    def _multiply_into(product, terms_a, terms_b):
        """
        Adds the product of two polynomials given by _masks into product.
        Each monomial of the product is identified by the pair
        (a | b, a & b): its edges, and which of those are squared.
        """
        for a, c in terms_a:
            for b, d in terms_b:
                key = (a | b, a & b)
                product[key] = product.get(key, 0) + c * d

    @staticmethod
    def _terms(poly):
        """The terms of a polynomial as (coefficient, edge indices) pairs, for fast evaluation."""
        if not isinstance(poly, Polynomial):
            return []
        return [
            (c, tuple(e for e, k in enumerate(power) for _ in range(k)))
            for power, c in poly.coefficients.items()
        ]

    @staticmethod
    def _evaluate(terms, point):
        return sum(c * math.prod(point[e] for e in edges) for c, edges in terms)

    def get_numeric_equations(self, phi_P, phi_Q, sfps_P, sfps_Q, extra_points=10, seed=0):
        """
        Builds a linear system for the a(i,j) in

            phi_P * phi_Q = sum of a(i,j) * phi_P_i * phi_Q_j

        by evaluating both sides exactly at random integer points, with one
        equation per point, in the same (row, constant) form as get_equations.

        Each point gives a random combination of the equations of
        get_equations, so with a few more points than unknowns both systems
        have the same solutions, except with vanishingly small probability.
        Use verify_solution to confirm a solution symbolically.
        """
        num_points = len(sfps_P) * len(sfps_Q) + extra_points
        self.debug(f'Evaluating polynomials at {num_points} random points')
        rng = random.Random(seed)
        terms_lhs = [self._terms(phi_P), self._terms(phi_Q)]
        terms_P = [self._terms(poly) for _, poly in sfps_P]
        terms_Q = [self._terms(poly) for _, poly in sfps_Q]

        equations = []
        for _ in range(num_points):
            point = [rng.randint(1, 1000) for _ in range(self.num_edges)]
            values_P = [self._evaluate(terms, point) for terms in terms_P]
            values_Q = [self._evaluate(terms, point) for terms in terms_Q]
            row = tuple(p * q for p in values_P for q in values_Q)
            constant = math.prod(self._evaluate(terms, point) for terms in terms_lhs)
            equations.append((row, constant))
        return equations

    def verify_solution(self, P, Q, solution, reduce=False):
        """
        Confirms symbolically that every member of a solution family returned
        by solve_problem satisfies the identity, using the exact system of
        get_equations.
        """
        phi_P, phi_Q = self.get_lhs(P, Q)
        sfps_P, sfps_Q = self.get_rhs(P, Q, reduce=reduce)
        return self._verify(phi_P, phi_Q, sfps_P, sfps_Q, solution)

    def _verify(self, phi_P, phi_Q, sfps_P, sfps_Q, solution):
        self.debug('Verifying solution symbolically')
        coefficients = [solution[(p, q)] for p, _ in sfps_P for q, _ in sfps_Q]
        for row, constant in self.get_equations(phi_P, phi_Q, sfps_P, sfps_Q):
            lhs = sum((r * c for r, c in zip(row, coefficients) if r), sym.Integer(0))
            if sym.expand(lhs - constant) != 0:
                return False
        return True

    def solve_system(self, equations, a_vars):
        self.debug(f'Solving linear system')
        xs = sym.symbols(f'x_:{len(a_vars)}')
        # Sort to keep the parametrization of the solution reproducible
        equations = sorted(equations)
        A = sym.Matrix([row for row, _ in equations])
        b = sym.Matrix([constant for _, constant in equations])
        return sym.linsolve((A, b), xs)

    def solve_problem(self, P, Q, reduce=False, verify=False):
        """
        Finds the family of identities phi_P * phi_Q = sum of a(i,j) * phi_P_k * phi_Q_k
        from random evaluations. With verify=True the family is also
        confirmed symbolically, which is much slower.
        """
        phi_P, phi_Q = self.get_lhs(P, Q)
        sfps_P, sfps_Q = self.get_rhs(P, Q, reduce=reduce)
        a_vars, reverse_variables, var_map = self.get_abstract_vars(sfps_P, sfps_Q)
        self.debug(f'We have {len(a_vars)} a_i,j variables')

        equations = self.get_numeric_equations(phi_P, phi_Q, sfps_P, sfps_Q)
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
        solution = {
            pair: values[a_vars[reverse_variables[coords]]] for pair, coords in var_map.items()
        }

        if verify:
            if not self._verify(phi_P, phi_Q, sfps_P, sfps_Q, solution):
                # Only possible if the random points were degenerate
                print('\nSymbolic verification FAILED')
                return None
            print('\nVerified symbolically')

        return solution

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
    # solver = Solver(get_kn(n))
    # solver.set_vertices(vertices)
    # P = '1' * (n-1)
    # Q = ''.join(map(lambda x: str(x+1), vertices))
    # solver.solve_problem(P, Q)

    # vertices = [0, 1, 2, 3, 4]
    # solver = Solver(get_kn(6))
    # solver.set_vertices(vertices)
    # P = '11111'
    # Q = '12345'
    # solver.solve_problem(P, Q, reduce=True)

    # vertices = [0, 1, 2]
    # solver = Solver(get_kn(5))
    # solver.set_vertices(vertices)
    # P = '111'
    # Q = '123'
    # solver.solve_problem(P, Q, reduce=True)

    # vertices = [0, 1, 2, 3, 4]
    # solver = Solver(get_kn(6))
    # solver.set_vertices(vertices)
    # P = '11122'
    # Q = '12345'
    # solver.solve_problem(P, Q, reduce=False)

    vertices = [0, 1, 2, 3]
    solver = Solver(get_kn(5))
    solver.set_vertices(vertices)
    P = '1111'
    Q = '1234'
    solver.solve_problem(P, Q, reduce=True)

    end = time()
    print(f'\nComputation took: {int(100*(end-start))/100} sec')
