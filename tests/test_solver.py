import logging
import random
from fractions import Fraction

import numpy as np
import pytest
import sympy

from helpers import GRAPHS, adjacency, assert_identity, make_solver
from spanning_forests.graph import Graph, complete_graph
from spanning_forests.linear.equations import _unique_runs
from spanning_forests.linear.evaluation import (
    MAX_POINT,
    crt,
    evaluate_all,
    evaluate_all_numpy,
    random_points,
)
from spanning_forests.linear.modular import (
    rational_reconstruction,
    reduced_row_echelon,
    solve_modular,
)
from spanning_forests.polynomial import MultilinearPolynomial
from spanning_forests.solution import Solution, classify

CASES = [
    # name, marked, P, Q, balanced, expected dimension of the solution space
    ('K4', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2], '112', '123', False, 0),
    ('K5', [0, 1, 2], '111', '122', False, 0),
    ('K5', [4, 0, 2], '111', '123', True, 3),
    ('K5', [0, 1, 2, 3], '1111', '1234', True, 8),
    ('PRISM', [0, 1, 2, 3], '1111', '1234', True, 8),
]


def solve(name, marked, P, Q, balanced):
    solver = make_solver(GRAPHS[name], marked)
    return solver, solver.solve_problem(P, Q, balanced)


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_solution_family_satisfies_identity(name, marked, P, Q, balanced, dimension):
    solver, solution = solve(name, marked, P, Q, balanced)

    assert solution.dimension == dimension

    # Every member of the family should be an identity, so check a few
    rng = random.Random(0)
    for _ in range(3):
        params = {x: rng.randint(-5, 5) for x in solution.free_parameters}
        rhs = {pair: int(c.subs(params)) for pair, c in solution.coefficients.items()}
        assert_identity(solver, (P, Q), rhs)


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_classification_covers_every_coefficient(name, marked, P, Q, balanced, dimension):
    _, solution = solve(name, marked, P, Q, balanced)
    classes = solution.classify()

    assert sum(len(c) for c in classes.values()) == len(solution.coefficients)
    assert len(classes['free']) == dimension


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_numeric_and_symbolic_systems_agree(name, marked, P, Q, balanced, dimension):
    solver = make_solver(GRAPHS[name], marked)
    problem = solver.problem(P, Q, balanced)
    num_vars = len(problem.pairs)

    numeric = solver.get_numeric_equations(problem)
    symbolic = solver.get_equations(problem)
    assert solver.solve_system(numeric, num_vars) == solver.solve_system(symbolic, num_vars)


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_modular_and_exact_solvers_agree(name, marked, P, Q, balanced, dimension):
    solver = make_solver(GRAPHS[name], marked)
    problem = solver.problem(P, Q, balanced)
    num_vars = len(problem.pairs)

    for equations in (solver.get_numeric_equations(problem), solver.get_equations(problem)):
        exact = solver.solve_system(equations, num_vars, modular=False)
        assert solver.solve_system(equations, num_vars, modular=True) == exact


@pytest.mark.parametrize('m', [3, 4, 5, 6])
def test_dimension_for_any_number_of_marked_vertices(m):
    # Theorem 3.4.1 of Fraser and Yeats
    solver = make_solver(complete_graph(m + 1), list(range(m)))
    P, Q = '1' * m, ''.join(str(i + 1) for i in range(m))
    assert solver.solve_problem(P, Q).dimension == m * (m - 2)


def test_modular_solver_with_no_solution():
    assert solve_modular([((1, 2), 3), ((2, 4), 7)], 2) is None


def test_modular_solver_with_free_parameters_and_fractions():
    x = sympy.symbols('x_:3')
    equations = [((2, 0, 1), 1), ((4, 0, 2), 2)]
    assert solve_modular(equations, 3) == (sympy.Rational(1, 2) - x[2] / 2, x[1], x[2])


def test_modular_solver_with_more_digits_than_one_prime():
    numerator, denominator = 2**70 + 1, 3**30
    solution = solve_modular([((denominator,), numerator)], 1)
    assert solution == (sympy.Rational(numerator, denominator),)


def test_modular_solver_discards_an_unlucky_prime():
    # Modulo the first prime, the first equation disappears
    p = 2**31 - 1
    assert solve_modular([((p, 0), p), ((0, 1), 1)], 2) == (1, 1)
    assert solve_modular([((p, 1), p + 1), ((0, 2), 2), ((p, 3), p + 3)], 2) == (1, 1)


def test_modular_solver_with_no_equations():
    assert solve_modular([], 2) == sympy.symbols('x_:2')


def test_reduced_row_echelon():
    matrix = np.array([[0, 2, 4], [0, 1, 2], [3, 0, 1]], dtype=np.int64)
    reduced, pivots = reduced_row_echelon(matrix, 7)
    assert pivots == (0, 1)
    assert reduced.tolist() == [[1, 0, 5], [0, 1, 2]]


@pytest.mark.parametrize('n, d', [(0, 1), (1, 1), (-1, 1), (3, 7), (-22, 41), (1000, 1001)])
def test_rational_reconstruction(n, d):
    modulus = 2**31 - 1
    value = n * pow(d, -1, modulus) % modulus
    assert rational_reconstruction(value, modulus) == Fraction(n, d)


def test_rational_reconstruction_fails_without_enough_digits():
    assert rational_reconstruction(5, 7) is None


@pytest.mark.parametrize(
    'name, marked, P, Q', [('K5', [0, 1, 2], '111', '123'), ('PRISM', [0, 1, 2, 3], '1111', '1234')]
)
def test_chunking_and_numpy_do_not_change_equations(name, marked, P, Q):
    solver = make_solver(GRAPHS[name], marked)
    problem = solver.problem(P, Q, balanced=True)

    expected = solver.get_equations(problem, chunk_edges=0, use_numpy=False)
    for use_numpy in (False, True):
        for chunk_edges in (0, 2, 6, 8):
            assert solver.get_equations(problem, chunk_edges, use_numpy) == expected


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_numpy_evaluation_gives_the_same_numeric_equations(name, marked, P, Q, balanced, dimension):
    solver = make_solver(GRAPHS[name], marked)
    problem = solver.problem(P, Q, balanced)
    expected = solver.get_numeric_equations(problem, use_numpy=False)
    assert solver.get_numeric_equations(problem, use_numpy=True) == expected


def test_numpy_evaluation_with_negative_and_large_coefficients():
    variables = ['x', 'y', 'z']
    # The terms -3xy + 10**30 yz - 10**25 + 7x + xyz
    terms = {0b011: -3, 0b110: 10**30, 0b000: -(10**25), 0b001: 7, 0b111: 1}
    poly = MultilinearPolynomial(variables, terms)
    points = [[1, 2, 3], [1000, 999, 998], [5, 1, 1000]]
    assert evaluate_all_numpy(poly, points) == evaluate_all(poly, points)
    assert evaluate_all(poly, points)[0] == -3 * 2 + 10**30 * 6 - 10**25 + 7 + 6
    assert evaluate_all_numpy(MultilinearPolynomial(variables, {}), points) == [0, 0, 0]


def test_numpy_evaluation_of_a_long_path_at_the_largest_coordinates():
    # The single spanning tree has 40 edges, so its values have 800 bits
    graph = Graph(adjacency(41, [(i, i + 1) for i in range(40)]))
    points = [[MAX_POINT] * 40, *random_points(3, 40)]
    values = evaluate_all_numpy(graph.tree_poly, points)
    assert values == evaluate_all(graph.tree_poly, points)
    assert values[0] == MAX_POINT**40


def runs_of(columns, coeffs, runs):
    return {(tuple(columns[s : s + n]), tuple(coeffs[s : s + n])) for s, n in runs}


def test_unique_runs(monkeypatch):
    # Runs: [(0, 1), (2, 3)], [(1, 5)], [(0, 1), (2, 3)], [(0, 1), (2, 4)], [(1, 5)]
    columns = np.array([0, 2, 1, 0, 2, 0, 2, 1])
    coeffs = np.array([1, 3, 5, 1, 3, 1, 4, 5])
    starts = np.array([0, 2, 3, 5, 7])
    lengths = np.array([2, 1, 2, 2, 1])
    expected = {((0, 2), (1, 3)), ((1,), (5,)), ((0, 2), (1, 4))}

    runs = list(_unique_runs(starts, lengths, columns, coeffs))
    assert len(runs) == 3
    assert runs_of(columns, coeffs, runs) == expected

    # With every hash colliding, the exact comparison takes over
    class ZeroWeights:
        def integers(self, low, high, size, dtype):
            return np.zeros(size, dtype=dtype)

    monkeypatch.setattr(np.random, 'default_rng', lambda seed: ZeroWeights())
    runs = list(_unique_runs(starts, lengths, columns, coeffs))
    assert len(runs) == 3
    assert runs_of(columns, coeffs, runs) == expected


def test_crt():
    primes = [7, 11, 13]
    for value in (0, 1, 500, 1000):
        assert crt([value % p for p in primes], primes) == value


def test_numpy_equations_with_coefficients_other_than_one():
    solver = make_solver(GRAPHS['K5'], [0, 1, 2])
    problem = solver.problem('111', '123')
    problem.phi_P = problem.phi_P * 2
    expected = solver.get_equations(problem, use_numpy=False)
    assert solver.get_equations(problem, use_numpy=True) == expected


def test_numpy_equations_with_empty_polynomials():
    # In a disconnected graph phi_P is 0, so the identity has no solution
    solver = make_solver(adjacency(5, [(0, 1), (1, 2), (3, 4)]), [0, 1, 2])
    problem = solver.problem('111', '123', balanced=True)
    expected = solver.get_equations(problem, use_numpy=False)
    assert solver.get_equations(problem, use_numpy=True) == expected


@pytest.mark.parametrize('name, marked, P, Q, balanced, dimension', CASES)
def test_verify_solution(name, marked, P, Q, balanced, dimension):
    solver, solution = solve(name, marked, P, Q, balanced)
    assert solution is not None
    assert solver.verify_solution(solution)

    # Changing any nonzero coefficient breaks the identity
    coefficients = solution.coefficients
    pair = next(pair for pair, c in coefficients.items() if c != 0)
    tampered = Solution((P, Q), {**coefficients, pair: coefficients[pair] + 1})
    assert not solver.verify_solution(tampered)


def test_solve_problem_with_verify():
    solver = make_solver(GRAPHS['K5'], [0, 1, 2, 3])
    assert solver.solve_problem('1111', '1234', balanced=True, verify=True).verified is True


def test_classify_solution():
    x, y = sympy.symbols('x y')
    values = {'a': 0, 'b': x, 'c': 1, 'd': -2, 'e': 1 - x, 'f': 2 * y}
    classes = classify(values)
    assert {kind: [key for key, _ in entries] for kind, entries in classes.items()} == {
        'forbidden': ['a'],
        'free': ['b'],
        'fixed': ['c', 'd'],
        'dependent': ['e', 'f'],
    }


def test_evaluation_primes_are_the_largest_primes_below_2_31():
    from spanning_forests.linear.evaluation import EVALUATION_PRIMES

    assert EVALUATION_PRIMES == sorted(EVALUATION_PRIMES, reverse=True)
    expected = [sympy.prevprime(2**31)]
    while len(expected) < len(EVALUATION_PRIMES):
        expected.append(sympy.prevprime(expected[-1]))
    assert EVALUATION_PRIMES == expected


def test_progress_is_logged(caplog):
    solver = make_solver(GRAPHS['K5'], [0, 1, 2])
    with caplog.at_level(logging.INFO, logger='identities'):
        solver.solve_problem('111', '123', balanced=True)
    assert 'Solving for 9 unknowns' in caplog.text


def test_numpy_evaluation_rejects_values_too_large_for_the_primes():
    poly = MultilinearPolynomial(['x'], {0b1: 10**300})
    with pytest.raises(ValueError, match='too large'):
        evaluate_all_numpy(poly, [[1]])
