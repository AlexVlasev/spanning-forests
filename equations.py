"""
The exact linear system for the unknowns of an identity search,

    phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

from expanding every product of spanning forest polynomials, with one
equation per monomial. Used to verify solutions symbolically. Products with
any number of factors are handled by expansion_equations.
"""

import itertools
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from aliases import Equation, IntArray, UIntArray
from polynomial import MultilinearPolynomial

if TYPE_CHECKING:
    from identities import Problem

# Monomial keys sum exponent * 3**e over edges e, and products of monomials
# without squares have keys below 3**num_edges, which fits in a uint64 up to 40 edges
MAX_NUMPY_EDGES = 40

type KeyArrays = tuple[UIntArray, IntArray]
type Terms = list[tuple[int, int]]
type Buckets = dict[int, Terms]
type ArrayBuckets = dict[int, KeyArrays]
type SparseEquation = tuple[tuple[tuple[int, int], ...], int]


def exact_equations(
    problem: 'Problem', num_edges: int, chunk_edges: int = 8, use_numpy: bool = True
) -> set[Equation]:
    """
    Builds the linear system for the unknowns c(P_i, Q_j) of a Problem, with
    one equation per edge monomial, saying that its coefficients on both
    sides agree. Each equation is returned as a pair (row, constant) where
    row holds the coefficient of each unknown, in the order of
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
    chunk_mask = (1 << min(chunk_edges, num_edges)) - 1
    use_numpy = use_numpy and num_edges <= MAX_NUMPY_EDGES

    # Factor 0 is the left hand side, and factor k + 1 multiplies unknown number k,
    # with the terms of each polynomial as (bitmask of edges, coefficient) pairs
    terms_P = [list(poly.terms.items()) for poly in problem.rhs_P.values()]
    terms_Q = [list(poly.terms.items()) for poly in problem.rhs_Q.values()]
    factors = [(list(problem.phi_P.terms.items()), list(problem.phi_Q.terms.items()))]
    factors.extend(itertools.product(terms_P, terms_Q))

    chunks = dict()
    for x in range(chunk_mask + 1):
        for y in range(chunk_mask + 1):
            chunks.setdefault((x | y, x & y), []).append((x, y))

    equations: set[Equation] = set()
    if use_numpy:
        array_buckets = [
            (_bucket_arrays(a, num_edges, chunk_mask), _bucket_arrays(b, num_edges, chunk_mask))
            for a, b in factors
        ]
        for pairs in chunks.values():
            equations.update(_chunk_equations_numpy(array_buckets, pairs))
    else:
        buckets = [(_bucket(a, chunk_mask), _bucket(b, chunk_mask)) for a, b in factors]
        for pairs in chunks.values():
            equations.update(_dense(_chunk_equations(buckets, pairs), len(factors) - 1))
    return equations


def expansion_equations(
    lhs: list[MultilinearPolynomial],
    terms: list[list[MultilinearPolynomial]],
    num_edges: int,
    block_size: int = 2**24,
) -> set[Equation]:
    """
    The exact linear system saying that the product of the polynomials in lhs
    is a combination of products of polynomials, with one unknown for each
    list of polynomials in terms. Equations have the form of exact_equations.

    Monomials are keyed as in _bucket_arrays, in base one more than the
    largest number of factors, which must fit in a uint64. Each product is
    built one factor at a time, in blocks of block_size pairs of terms. The
    arithmetic is exact: a ValueError is raised if a coefficient of a product
    might not fit in an int64, which the sum of the absolute values of the
    coefficients of its factors would bound.
    """
    base = max(len(lhs), *map(len, terms)) + 1 if terms else len(lhs) + 1
    if base**num_edges > 2**64:
        raise ValueError(f'too many edges to expand products of {base - 1} factors')

    arrays = dict()
    keys, columns, coeffs = [], [], []
    for column, factors in enumerate([lhs, *terms]):
        for poly in factors:
            if id(poly) not in arrays:
                arrays[id(poly)] = _key_arrays(poly, num_edges, base)
        if math.prod(int(np.abs(arrays[id(poly)][1]).sum()) for poly in factors) >= 2**63:
            raise ValueError(f'coefficients too large to expand products of {len(factors)} factors')
        column_keys, column_coeffs = _expand_product(
            sorted((arrays[id(poly)] for poly in factors), key=lambda a: len(a[0])), block_size
        )
        keys.append(column_keys)
        coeffs.append(column_coeffs)
        columns.append(np.full(len(column_keys), column, dtype=np.int64))
    return _equations_from_columns(keys, columns, coeffs, len(terms))


def _key_arrays(poly: MultilinearPolynomial, num_edges: int, base: int) -> KeyArrays:
    """The numpy arrays of the monomial keys and coefficients of a polynomial."""
    masks = np.array(list(poly.terms), dtype=np.uint64)
    coeffs = np.array(list(poly.terms.values()), dtype=np.int64)
    return _monomial_keys(masks, num_edges, base), coeffs


def _expand_product(factors: list[KeyArrays], block_size: int) -> KeyArrays:
    """The distinct monomial keys and coefficients of a product of (keys, coefficients)."""
    keys = np.zeros(1, dtype=np.uint64)
    coeffs = np.ones(1, dtype=np.int64)
    for factor_keys, factor_coeffs in factors:
        step = max(1, block_size // max(1, len(factor_keys)))
        total_keys = np.zeros(0, dtype=np.uint64)
        total_coeffs = np.zeros(0, dtype=np.int64)
        for start in range(0, len(keys), step):
            block_keys = np.add.outer(keys[start : start + step], factor_keys)
            block_coeffs = np.multiply.outer(coeffs[start : start + step], factor_coeffs)
            total_keys, total_coeffs = _sum_by_key(
                np.concatenate((total_keys, block_keys.ravel())),
                np.concatenate((total_coeffs, block_coeffs.ravel())),
            )
        keys, coeffs = total_keys, total_coeffs
    return keys, coeffs


def _chunk_equations(
    buckets: list[tuple[Buckets, Buckets]], pairs: list[tuple[int, int]]
) -> set[SparseEquation]:
    """
    The equations of exact_equations for the monomials in one chunk, with
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


def _dense(sparse_equations: set[SparseEquation], num_vars: int) -> set[Equation]:
    """Converts equations with sparse rows to the dense rows of exact_equations."""
    equations = set()
    for entries, constant in sparse_equations:
        row = [0] * num_vars
        for k, c in entries:
            row[k] = c
        equations.add((tuple(row), constant))
    return equations


def _bucket(terms: Terms, mask: int) -> Buckets:
    """Sorts (bitmask, coefficient) terms by which of the edges in mask they contain."""
    buckets = dict()
    for term in terms:
        buckets.setdefault(term[0] & mask, []).append(term)
    return buckets


def _bucket_arrays(terms: Terms, num_edges: int, mask: int) -> ArrayBuckets:
    """
    Like _bucket, but each bucket holds numpy arrays of the monomial keys and
    coefficients of its terms. The key of a monomial is the sum of
    exponent * 3**e over its edges e, so the key of a product of two
    monomials without squares is the sum of their keys.
    """
    masks = np.array([m for m, _ in terms], dtype=np.uint64)
    coeffs = np.array([c for _, c in terms], dtype=np.int64)
    keys = _monomial_keys(masks, num_edges)

    ids = masks & np.uint64(mask)
    order = np.argsort(ids, kind='stable')
    ids, keys, coeffs = ids[order], keys[order], coeffs[order]
    values, starts = np.unique(ids, return_index=True)
    ends = [*starts[1:], len(ids)]
    return {int(v): (keys[s:e], coeffs[s:e]) for v, s, e in zip(values, starts, ends)}


def _monomial_keys(masks: UIntArray, num_edges: int, base: int = 3) -> UIntArray:
    """The key of each monomial without squares, the sum of base**e over its edges e."""
    keys = np.zeros(len(masks), dtype=np.uint64)
    for e in range(num_edges):
        keys += ((masks >> np.uint64(e)) & np.uint64(1)) * np.uint64(base**e)
    return keys


def _chunk_equations_numpy(
    buckets: list[tuple[ArrayBuckets, ArrayBuckets]], pairs: list[tuple[int, int]]
) -> set[Equation]:
    """
    The equations of exact_equations for the monomials in one chunk, from the
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
    return _equations_from_columns(keys, columns, coeffs, len(buckets) - 1)


def _equations_from_columns(
    key_arrays: list[UIntArray],
    column_arrays: list[IntArray],
    coeff_arrays: list[IntArray],
    num_vars: int,
) -> set[Equation]:
    """
    The equations of exact_equations from arrays of the monomial keys and
    coefficients in each column, with the constant in column 0.
    """
    # Sparse rows, as runs of (column, coefficient) entries grouped by monomial,
    # with the constant in column 0, and the coefficient of unknown number k in
    # column k + 1. Columns were added in order, so a stable sort keeps them sorted
    _, inverse = np.unique(np.concatenate(key_arrays), return_inverse=True)
    order = np.argsort(inverse.ravel(), kind='stable')
    monomials = inverse.ravel()[order]
    columns = np.concatenate(column_arrays)[order]
    coeffs = np.concatenate(coeff_arrays)[order]

    nonzero = coeffs != 0
    monomials, columns, coeffs = monomials[nonzero], columns[nonzero], coeffs[nonzero]
    if not len(coeffs):
        return set()
    starts = np.flatnonzero(np.concatenate(([True], monomials[1:] != monomials[:-1])))
    lengths = np.diff(np.append(starts, len(coeffs)))
    coeffs = coeffs // np.repeat(np.gcd.reduceat(np.abs(coeffs), starts), lengths)

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


def _unique_runs(
    starts: IntArray, lengths: IntArray, columns: IntArray, coeffs: IntArray
) -> Iterable[tuple[int, int]]:
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


def _sum_by_key(keys: UIntArray, coeffs: IntArray) -> KeyArrays:
    """The distinct keys, sorted, and the sum of the coefficients of each."""
    if np.all(coeffs == 1):
        # Only needs a sort rather than an argsort
        return np.unique(keys, return_counts=True)
    distinct, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros(len(distinct), dtype=np.int64)
    np.add.at(sums, inverse.ravel(), coeffs)
    return distinct, sums


def _multiply_into(
    product: dict[tuple[int, int], int],
    terms_a: Iterable[tuple[int, int]],
    terms_b: Iterable[tuple[int, int]],
) -> None:
    """
    Adds the product of two polynomials, given as (bitmask of edges,
    coefficient) pairs, into product.
    Each monomial of the product is identified by the pair
    (a | b, a & b): its edges, and which of those are squared.
    """
    for a, c in terms_a:
        for b, d in terms_b:
            key = (a | b, a & b)
            product[key] = product.get(key, 0) + c * d

