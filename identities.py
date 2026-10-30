"""
Searches for quadratic identities between spanning forest polynomials,

    phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

where P_i and Q_j range over all partitions of the marked vertices into a
fixed number of parts. Identities are found from exact evaluations at random
points, and can then be verified symbolically.
"""

import itertools
import math
import random

import numpy as np
import sympy as sym

from solution import Solution

# Monomial keys sum exponent * 3**e over edges e, and products of monomials
# without squares have keys below 3**num_edges, which fits in a uint64 up to 40 edges
MAX_NUMPY_EDGES = 40

# Random points have coordinates from 1 to MAX_POINT
MAX_POINT = 1000

# Primes below 2**31 for evaluating modulo p, so that products of two residues fit in an int64
EVALUATION_PRIMES = [2**31 - 1]
while len(EVALUATION_PRIMES) < 16:
    EVALUATION_PRIMES.append(sym.prevprime(EVALUATION_PRIMES[-1]))


class Problem:
    """
    The polynomials of a search for identities

        phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

    with the spanning forest polynomials of the P_i and Q_j in rhs_P and
    rhs_Q, keyed by pattern. The unknowns c are ordered as in pairs.
    """

    def __init__(self, P, Q, phi_P, phi_Q, rhs_P, rhs_Q):
        self.P = P
        self.Q = Q
        self.phi_P = phi_P
        self.phi_Q = phi_Q
        self.rhs_P = rhs_P
        self.rhs_Q = rhs_Q
        self.pairs = [(p, q) for p in rhs_P for q in rhs_Q]


class IdentitySolver:
    def __init__(self, graph, marked, verbose=False):
        self.graph = graph
        self.marked = marked
        self.verbose = verbose

    def problem(self, P, Q, reduce=False):
        """
        The Problem for P and Q, where by default the P_i have one part more
        than P and the Q_j one part fewer than Q. With reduce=True, the
        numbers of parts are balanced instead, with the P_i having the fewer.
        """
        p, q = len(set(P)), len(set(Q))
        if reduce:
            return self._problem(P, Q, (p + q) // 2, (p + q + 1) // 2)
        return self._problem(P, Q, p + 1, q - 1)

    def _problem(self, P, Q, parts_P, parts_Q):
        phi_P = self.graph.forest_polynomial(self.marked, P)
        phi_Q = self.graph.forest_polynomial(self.marked, Q)
        self.debug(
            f'Polynomials {P} and {Q} have {len(phi_P.coefficients)} and '
            f'{len(phi_Q.coefficients)} terms'
        )

        rhs_P = self.graph.forest_polynomials(self.marked, parts_P)
        rhs_Q = self.graph.forest_polynomials(self.marked, parts_Q)
        self.debug(
            f'Found {len(rhs_P)} P_i with {parts_P} parts and '
            f'{len(rhs_Q)} Q_j with {parts_Q} parts'
        )

        return Problem(P, Q, phi_P, phi_Q, rhs_P, rhs_Q)

    def get_equations(self, problem, chunk_edges=8, use_numpy=True):
        """
        Builds the linear system for the unknowns c(P_i, Q_j) of a Problem,

            phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

        with one equation per edge monomial, saying that its coefficients on
        both sides agree. Each equation is returned as a pair (row, constant)
        where row holds the coefficient of each unknown, in the order of
        Problem.pairs, reduced by their gcd. Duplicates are dropped.

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
        self.debug(f'Multiplying {len(problem.pairs) + 1} pairs of polynomials')
        chunk_mask = (1 << min(chunk_edges, self.graph.num_edges)) - 1
        use_numpy = use_numpy and self.graph.num_edges <= MAX_NUMPY_EDGES

        # Factor 0 is the left hand side, and factor k + 1 multiplies unknown number k
        masks_P = [_masks(poly) for poly in problem.rhs_P.values()]
        masks_Q = [_masks(poly) for poly in problem.rhs_Q.values()]
        factors = [(_masks(problem.phi_P), _masks(problem.phi_Q))]
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

    def get_numeric_equations(self, problem, extra_points=10, seed=0, use_numpy=True):
        """
        Builds a linear system for the unknowns of a Problem by evaluating
        both sides exactly at random integer points, with one equation per
        point, in the same (row, constant) form as get_equations.

        Each point gives a random combination of the equations of
        get_equations, so with a few more points than unknowns both systems
        have the same solutions, except with vanishingly small probability.
        Use verify_solution to confirm a solution symbolically.

        With use_numpy, the polynomials are evaluated at all points at once,
        modulo a few primes, and the exact values recovered with the Chinese
        remainder theorem. Otherwise they are evaluated one point at a time.
        """
        num_points = len(problem.pairs) + extra_points
        self.debug(f'Evaluating polynomials at {num_points} random points')
        rng = random.Random(seed)
        points = [
            [rng.randint(1, MAX_POINT) for _ in range(self.graph.num_edges)]
            for _ in range(num_points)
        ]

        evaluate = _evaluate_all_numpy if use_numpy else _evaluate_all
        values_lhs = [evaluate(poly, points) for poly in (problem.phi_P, problem.phi_Q)]
        values_P = [evaluate(poly, points) for poly in problem.rhs_P.values()]
        values_Q = [evaluate(poly, points) for poly in problem.rhs_Q.values()]

        equations = []
        for i in range(num_points):
            row = tuple(p[i] * q[i] for p in values_P for q in values_Q)
            constant = values_lhs[0][i] * values_lhs[1][i]
            equations.append((row, constant))
        return equations

    def verify_solution(self, solution):
        """
        Confirms symbolically that every member of a solution family returned
        by solve_problem satisfies the identity, using the exact system of
        get_equations.
        """
        # The numbers of parts of the P_i and Q_j, from any pair of patterns
        p, q = next(iter(solution.coefficients))
        problem = self._problem(solution.P, solution.Q, len(set(p)), len(set(q)))
        return self._verify(problem, solution)

    def _verify(self, problem, solution):
        self.debug('Verifying solution symbolically')
        coefficients = [solution.coefficients[pair] for pair in problem.pairs]
        for row, constant in self.get_equations(problem):
            lhs = sum((r * c for r, c in zip(row, coefficients) if r), sym.Integer(0))
            if sym.expand(lhs - constant) != 0:
                return False
        return True

    def solve_system(self, equations, num_vars):
        """
        The general solution of a linear system of (row, constant) equations,
        as a tuple of sympy expressions in free parameters x_0, x_1, ..., or
        None if there is no solution.
        """
        self.debug(f'Solving linear system')
        xs = sym.symbols(f'x_:{num_vars}')
        # Sort to keep the parametrization of the solution reproducible
        equations = sorted(equations)
        A = sym.Matrix([row for row, _ in equations])
        b = sym.Matrix([constant for _, constant in equations])
        solutions = sym.linsolve((A, b), xs)
        # linsolve returns either EmptySet or a FiniteSet with one parametric solution
        return solutions.args[0].args if solutions.args else None

    def solve_problem(self, P, Q, reduce=False, verify=False):
        """
        Finds the family of identities phi_P * phi_Q = sum of c * phi_P_i * phi_Q_j
        from random evaluations, as a Solution, or None if there is none.
        With verify=True the family is also confirmed symbolically, which is
        much slower, and the result stored in Solution.verified.
        """
        problem = self.problem(P, Q, reduce)
        self.debug(f'Solving for {len(problem.pairs)} unknowns')

        values = self.solve_system(self.get_numeric_equations(problem), len(problem.pairs))
        if values is None:
            return None

        solution = Solution(P, Q, dict(zip(problem.pairs, values)))
        if verify:
            # Can only fail if the random points were degenerate
            solution.verified = self._verify(problem, solution)
        return solution

    def debug(self, expr):
        if self.verbose:
            print(expr)


def _chunk_equations(buckets, pairs):
    """
    The equations of get_equations for the monomials in one chunk, with
    sparse rows of (index of the unknown, coefficient) pairs.
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
    # Reduce each column to its distinct monomials before combining them, which
    # keeps only one column's products in memory at a time
    keys, columns, coeffs = [], [], []
    for column, (buckets_a, buckets_b) in enumerate(buckets):
        column_keys, column_coeffs = [], []
        for x, y in pairs:
            if x not in buckets_a or y not in buckets_b:
                continue
            keys_a, coeffs_a = buckets_a[x]
            keys_b, coeffs_b = buckets_b[y]
            column_keys.append(np.add.outer(keys_a, keys_b).ravel())
            column_coeffs.append(np.multiply.outer(coeffs_a, coeffs_b).ravel())
        if not column_keys:
            continue
        monomials, sums = _sum_by_key(np.concatenate(column_keys), np.concatenate(column_coeffs))
        keys.append(monomials)
        coeffs.append(sums)
        columns.append(np.full(len(monomials), column, dtype=np.int64))
    if not keys:
        return set()

    # Sparse rows, as runs of (column, coefficient) entries grouped by monomial,
    # with the constant in column 0, and the coefficient of unknown number k in
    # column k + 1. Columns were added in order, so a stable sort keeps them sorted
    _, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    order = np.argsort(inverse.ravel(), kind='stable')
    monomials = inverse.ravel()[order]
    columns = np.concatenate(columns)[order]
    coeffs = np.concatenate(coeffs)[order]

    nonzero = coeffs != 0
    monomials, columns, coeffs = monomials[nonzero], columns[nonzero], coeffs[nonzero]
    if not len(coeffs):
        return set()
    starts = np.flatnonzero(np.concatenate(([True], monomials[1:] != monomials[:-1])))
    lengths = np.diff(np.append(starts, len(coeffs)))
    coeffs = coeffs // np.repeat(np.gcd.reduceat(np.abs(coeffs), starts), lengths)

    num_vars = len(buckets) - 1
    equations = set()
    for start, length in _unique_runs(starts, lengths, columns, coeffs):
        row = [0] * num_vars
        constant = 0
        for column, c in zip(columns[start : start + length], coeffs[start : start + length]):
            if column == 0:
                constant = int(c)
            else:
                row[column - 1] = int(c)
        equations.add((tuple(row), constant))
    return equations


def _unique_runs(starts, lengths, columns, coeffs):
    """
    The (start, length) of one representative of each distinct run of
    (column, coefficient) entries. Runs are grouped by a hash, and kept as is
    after checking that every run equals the representative of its group.
    Otherwise, the runs are compared exactly instead.
    """
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 2**62, size=int(columns.max()) + 1, dtype=np.int64)
    with np.errstate(over='ignore'):
        hashes = np.add.reduceat(coeffs * weights[columns], starts)
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    representative = first[inverse.ravel()]

    if np.array_equal(lengths, lengths[representative]):
        # Compare every entry with the entry at the same offset in the representative run
        offsets = np.arange(len(coeffs)) - np.repeat(starts, lengths)
        matching = np.repeat(starts[representative], lengths) + offsets
        if np.array_equal(columns, columns[matching]) and np.array_equal(coeffs, coeffs[matching]):
            return zip(starts[first], lengths[first])

    runs = dict()
    for start, length in zip(starts, lengths):
        run = (tuple(columns[start : start + length]), tuple(coeffs[start : start + length]))
        runs.setdefault(run, (start, length))
    return runs.values()


def _sum_by_key(keys, coeffs):
    """The distinct keys, sorted, and the sum of the coefficients of each."""
    if np.all(coeffs == 1):
        # Only needs a sort rather than an argsort
        return np.unique(keys, return_counts=True)
    distinct, inverse = np.unique(keys, return_inverse=True)
    # Weights are summed as floats, which is exact for these small integers
    sums = np.rint(np.bincount(inverse.ravel(), weights=coeffs)).astype(np.int64)
    return distinct, sums


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


def _evaluate_all(poly, points):
    """The exact values of a polynomial at each point, one point at a time."""
    terms = _terms(poly)
    return [_evaluate(terms, point) for point in points]


def _evaluate_all_numpy(poly, points, block_size=2**22):
    """
    The exact values of a polynomial at each point. The polynomial is
    evaluated at all points at once modulo enough primes that their product
    exceeds twice the largest possible absolute value, and the exact values
    are recovered with the Chinese remainder theorem.
    """
    terms = _terms(poly)
    if not terms:
        return [0] * len(points)

    # Pad every term to the same degree with an extra coordinate equal to 1
    degree = max(len(edges) for _, edges in terms)
    num_edges = len(points[0])
    index = np.full((len(terms), degree), num_edges, dtype=np.int64)
    for t, (_, edges) in enumerate(terms):
        index[t, : len(edges)] = edges
    coordinates = np.array([[*point, 1] for point in points], dtype=np.int64)

    bound = 2 * sum(abs(c) for c, _ in terms) * max(map(max, points)) ** degree
    primes = []
    while math.prod(primes) <= bound:
        primes.append(EVALUATION_PRIMES[len(primes)])

    # Evaluate in blocks of terms to bound the size of the (points, terms) arrays
    step = max(1, block_size // len(points))
    residues = []
    for p in primes:
        coeffs = np.array([c % p for c, _ in terms], dtype=np.int64)
        total = np.zeros(len(points), dtype=np.int64)
        for start in range(0, len(terms), step):
            block = index[start : start + step]
            values = coeffs[start : start + step] * np.ones((len(points), 1), dtype=np.int64)
            for d in range(degree):
                values = values * coordinates[:, block[:, d]] % p
            total = (total + values.sum(axis=1)) % p
        residues.append(total)

    modulus = math.prod(primes)
    exact = []
    for i in range(len(points)):
        value = _crt([int(r[i]) for r in residues], primes)
        exact.append(value - modulus if value > modulus // 2 else value)
    return exact


def _crt(residues, primes):
    """The integer modulo the product of the primes with the given residues."""
    value, modulus = 0, 1
    for r, p in zip(residues, primes):
        value += modulus * ((r - value) * pow(modulus, -1, p) % p)
        modulus *= p
    return value
