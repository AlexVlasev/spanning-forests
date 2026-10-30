"""
Searches for quadratic identities between spanning forest polynomials,

    phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

where P_i and Q_j range over all partitions of the marked vertices into a
fixed number of parts. Identities are found from exact evaluations at random
points (see evaluation.py), and can then be verified symbolically by
expanding every product (see equations.py).
"""

import sympy as sym

from equations import exact_equations
from evaluation import numeric_equations
from solution import Solution


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
        """The exact linear system of a Problem, from equations.exact_equations."""
        self.debug(f'Multiplying {len(problem.pairs) + 1} pairs of polynomials')
        equations = exact_equations(problem, self.graph.num_edges, chunk_edges, use_numpy)
        self.debug(f'Found {len(equations)} distinct equations')
        return equations

    def get_numeric_equations(self, problem, extra_points=10, seed=0, use_numpy=True):
        """
        A linear system of a Problem from random evaluations, from
        evaluation.numeric_equations. Use verify_solution to confirm a
        solution symbolically.
        """
        self.debug(f'Evaluating polynomials at {len(problem.pairs) + extra_points} random points')
        return numeric_equations(problem, self.graph.num_edges, extra_points, seed, use_numpy)

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
