"""
Searches for quadratic identities between spanning forest polynomials,

    phi_P * phi_Q = sum of a(i,j) * phi_P_i * phi_Q_j

where P_i and Q_j range over all partitions of the marked vertices into a
fixed number of parts. Identities are found from exact evaluations at random
points, and can then be verified symbolically.
"""

import itertools
import math
import random

import numpy as np
import sympy as sym

# Monomial keys sum exponent * 3**e over edges e, and products of monomials
# without squares have keys below 3**num_edges, which fits in a uint64 up to 40 edges
MAX_NUMPY_EDGES = 40


class IdentitySolver:
    def __init__(self, graph, marked, verbose=False):
        self.graph = graph
        self.marked = marked
        self.verbose = verbose

    def get_lhs(self, P, Q):
        self.debug(f'Getting {P} polynomial')
        phi_P = self.graph.forest_polynomial(self.marked, P)
        self.debug(f'Polynomial {P} has {len(phi_P.coefficients.items())} terms')

        self.debug(f'Getting {Q} polynomial')
        phi_Q = self.graph.forest_polynomial(self.marked, Q)
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
        sfps_P = list(self.graph.forest_polynomials(self.marked, a).items())
        self.debug(f'Found {len(sfps_P)} P_k polynomials')

        self.debug(f'Getting Q_k polynomials. Partitions: {b}')
        sfps_Q = list(self.graph.forest_polynomials(self.marked, b).items())
        self.debug(f'Found {len(sfps_Q)} Q_k polynomials')

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

    def get_equations(self, phi_P, phi_Q, sfps_P, sfps_Q, chunk_edges=8, use_numpy=True):
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

        Graphs with up to MAX_NUMPY_EDGES edges multiply with numpy, unless
        use_numpy is False, and larger ones in pure Python.
        """
        self.debug(f'\nMultiplying P, Q polynomial pairs: {len(sfps_P) * len(sfps_Q) + 1}')
        chunk_mask = (1 << min(chunk_edges, self.graph.num_edges)) - 1
        use_numpy = use_numpy and self.graph.num_edges <= MAX_NUMPY_EDGES

        # Factor 0 is the left hand side, and factor k + 1 multiplies a(i,j) number k
        masks_P = [_masks(poly) for _, poly in sfps_P]
        masks_Q = [_masks(poly) for _, poly in sfps_Q]
        factors = [(_masks(phi_P), _masks(phi_Q))]
        factors.extend(itertools.product(masks_P, masks_Q))
        if use_numpy:
            bucket = lambda terms: _bucket_arrays(terms, self.graph.num_edges, chunk_mask)
        else:
            bucket = lambda terms: _bucket(terms, chunk_mask)
        buckets = [(bucket(a), bucket(b)) for a, b in factors]

        chunks = dict()
        for x in range(chunk_mask + 1):
            for y in range(chunk_mask + 1):
                chunks.setdefault((x | y, x & y), []).append((x, y))

        self.debug(f'Building equations in {len(chunks)} chunks')
        equations = set()
        for pairs in chunks.values():
            if use_numpy:
                equations.update(_chunk_equations_numpy(buckets, pairs))
            else:
                equations.update(_dense(_chunk_equations(buckets, pairs), len(factors) - 1))

        self.debug(f'Remaining equations: {len(equations)}')
        return equations

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
        terms_lhs = [_terms(phi_P), _terms(phi_Q)]
        terms_P = [_terms(poly) for _, poly in sfps_P]
        terms_Q = [_terms(poly) for _, poly in sfps_Q]

        equations = []
        for _ in range(num_points):
            point = [rng.randint(1, 1000) for _ in range(self.graph.num_edges)]
            values_P = [_evaluate(terms, point) for terms in terms_P]
            values_Q = [_evaluate(terms, point) for terms in terms_Q]
            row = tuple(p * q for p in values_P for q in values_Q)
            constant = math.prod(_evaluate(terms, point) for terms in terms_lhs)
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
        if self.verbose:
            print(expr)


def _chunk_equations(buckets, pairs):
    """
    The equations of get_equations for the monomials in one chunk, with
    sparse rows of (index of a(i,j), coefficient) pairs.
    """
    products = []
    for buckets_a, buckets_b in buckets:
        product = dict()
        for x, y in pairs:
            _multiply_into(product, buckets_a.get(x, ()), buckets_b.get(y, ()))
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


def _dense(sparse_equations, num_vars):
    """Converts equations with sparse rows to the dense rows of get_equations."""
    equations = set()
    for entries, constant in sparse_equations:
        row = [0] * num_vars
        for k, c in entries:
            row[k] = c
        equations.add((tuple(row), constant))
    return equations


def _bucket(terms, mask):
    """Sorts terms from _masks by which of the edges in mask they contain."""
    buckets = dict()
    for term in terms:
        buckets.setdefault(term[0] & mask, []).append(term)
    return buckets


def _bucket_arrays(terms, num_edges, mask):
    """
    Like _bucket, but each bucket holds numpy arrays of the monomial keys and
    coefficients of its terms. The key of a monomial is the sum of
    exponent * 3**e over its edges e, so the key of a product of two
    monomials without squares is the sum of their keys.
    """
    masks = np.array([m for m, _ in terms], dtype=np.uint64)
    coeffs = np.array([c for _, c in terms], dtype=np.int64)
    keys = np.zeros(len(terms), dtype=np.uint64)
    for e in range(num_edges):
        keys += ((masks >> np.uint64(e)) & np.uint64(1)) * np.uint64(3**e)

    ids = masks & np.uint64(mask)
    order = np.argsort(ids, kind='stable')
    ids, keys, coeffs = ids[order], keys[order], coeffs[order]
    values, starts = np.unique(ids, return_index=True)
    ends = [*starts[1:], len(ids)]
    return {int(v): (keys[s:e], coeffs[s:e]) for v, s, e in zip(values, starts, ends)}


def _chunk_equations_numpy(buckets, pairs):
    """
    The equations of get_equations for the monomials in one chunk, from the
    buckets of _bucket_arrays, with dense rows.
    """
    keys, columns, coeffs = [], [], []
    for column, (buckets_a, buckets_b) in enumerate(buckets):
        for x, y in pairs:
            if x not in buckets_a or y not in buckets_b:
                continue
            keys_a, coeffs_a = buckets_a[x]
            keys_b, coeffs_b = buckets_b[y]
            keys.append(np.add.outer(keys_a, keys_b).ravel())
            coeffs.append(np.multiply.outer(coeffs_a, coeffs_b).ravel())
            columns.append(np.full(keys[-1].size, column, dtype=np.uint64))
    if not keys:
        return set()

    # Sum the coefficients of each (monomial, column) into a dense matrix with
    # the constant in column 0, and the coefficient of a(i,j) number k in column k + 1
    keys = np.concatenate(keys)
    columns = np.concatenate(columns)
    coeffs = np.concatenate(coeffs)
    num_columns = len(buckets)
    if np.all(coeffs == 1) and int(keys.max()) * num_columns < 2**64:
        # Every coefficient is 1, so count the (monomial, column) pairs, packed
        # into one integer, which only needs a sort rather than an argsort
        pairs, counts = np.unique(keys * np.uint64(num_columns) + columns, return_counts=True)
        monomials = pairs // np.uint64(num_columns)
        starts = np.concatenate(([True], monomials[1:] != monomials[:-1]))
        rows = np.zeros((int(starts.sum()), num_columns), dtype=np.int64)
        rows[np.cumsum(starts) - 1, (pairs % np.uint64(num_columns)).astype(np.int64)] = counts
    else:
        monomials, inverse = np.unique(keys, return_inverse=True)
        index = inverse.ravel() * num_columns + columns.astype(np.int64)
        # Weights are summed as floats, which is exact for these small integers
        rows = np.rint(np.bincount(index, weights=coeffs, minlength=len(monomials) * num_columns))
        rows = rows.astype(np.int64).reshape(len(monomials), num_columns)

    gcd = np.gcd.reduce(rows, axis=1)
    rows = rows[gcd > 0] // gcd[gcd > 0, None]
    return {(tuple(int(c) for c in row[1:]), int(row[0])) for row in _unique_rows(rows)}


def _unique_rows(rows):
    """
    The distinct rows of a matrix. Equivalent to np.unique(rows, axis=0), but
    much faster for many rows: rows are grouped by a hash, and kept as is
    after checking that every row equals the first of its group.
    """
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 2**62, size=rows.shape[1], dtype=np.int64)
    with np.errstate(over='ignore'):
        hashes = rows @ weights
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    if np.array_equal(rows, rows[first[inverse.ravel()]]):
        return rows[first]
    return np.unique(rows, axis=0)


def _masks(poly):
    """
    The terms of a polynomial without squares, such as a spanning forest
    polynomial, as (bitmask of edges, coefficient) pairs.
    """
    terms = []
    for power, c in poly.coefficients.items():
        assert all(k <= 1 for k in power), 'Expected a polynomial without squares'
        terms.append((sum(1 << e for e, k in enumerate(power) if k), c))
    return terms


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


def _terms(poly):
    """The terms of a polynomial as (coefficient, edge indices) pairs, for fast evaluation."""
    return [
        (c, tuple(e for e, k in enumerate(power) for _ in range(k)))
        for power, c in poly.coefficients.items()
    ]


def _evaluate(terms, point):
    return sum(c * math.prod(point[e] for e in edges) for c, edges in terms)
