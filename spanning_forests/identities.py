"""
Searches for quadratic identities between spanning forest polynomials,

    phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

where P_i and Q_j range over all partitions of the marked vertices into a
fixed number of parts. Identities are found from exact evaluations at random
points (see evaluation.py), and can then be verified symbolically by
expanding every product (see equations.py).
"""

import logging
from collections.abc import Collection
from typing import cast

import sympy as sym

from spanning_forests.aliases import Equation, Marked, Pattern
from spanning_forests.graph import Graph
from spanning_forests.linear.equations import exact_equations
from spanning_forests.linear.evaluation import numeric_equations
from spanning_forests.linear.modular import solve_modular
from spanning_forests.polynomial import MultilinearPolynomial
from spanning_forests.solution import Solution

logger = logging.getLogger(__name__)


class Problem:
    """
    The polynomials of a search for identities

        phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j

    with the spanning forest polynomials of the P_i and Q_j in rhs_P and
    rhs_Q, keyed by pattern. The unknowns c are ordered as in pairs.
    """

    def __init__(
        self,
        P: Pattern,
        Q: Pattern,
        phi_P: MultilinearPolynomial,
        phi_Q: MultilinearPolynomial,
        rhs_P: dict[Pattern, MultilinearPolynomial],
        rhs_Q: dict[Pattern, MultilinearPolynomial],
    ) -> None:
        self.P = P
        self.Q = Q
        self.phi_P = phi_P
        self.phi_Q = phi_Q
        self.rhs_P = rhs_P
        self.rhs_Q = rhs_Q
        self.pairs = [(p, q) for p in rhs_P for q in rhs_Q]


class IdentitySolver:
    """Searches for identities between the forest polynomials of a graph with marked vertices."""

    def __init__(self, graph: Graph, marked: Marked) -> None:
        self.graph = graph
        self.marked = marked

    def problem(self, P: Pattern, Q: Pattern, balanced: bool = False) -> Problem:
        """
        The Problem for P and Q, where by default the P_i have one part more
        than P and the Q_j one part fewer than Q. With balanced=True, the
        numbers of parts are balanced instead, with the P_i having the fewer.
        """
        p, q = len(set(P)), len(set(Q))
        if balanced:
            return self._problem(P, Q, (p + q) // 2, (p + q + 1) // 2)
        return self._problem(P, Q, p + 1, q - 1)

    def _problem(self, P: Pattern, Q: Pattern, parts_P: int, parts_Q: int) -> Problem:
        phi_P = self.graph.forest_polynomial(self.marked, P)
        phi_Q = self.graph.forest_polynomial(self.marked, Q)
        logger.info(f'Polynomials {P} and {Q} have {len(phi_P)} and {len(phi_Q)} terms')

        rhs_P = self.graph.forest_polynomials(self.marked, parts_P)
        rhs_Q = self.graph.forest_polynomials(self.marked, parts_Q)
        logger.info(
            f'Found {len(rhs_P)} P_i with {parts_P} parts and '
            f'{len(rhs_Q)} Q_j with {parts_Q} parts'
        )

        return Problem(P, Q, phi_P, phi_Q, rhs_P, rhs_Q)

    def get_equations(
        self, problem: Problem, chunk_edges: int = 8, use_numpy: bool = True
    ) -> set[Equation]:
        """The exact linear system of a Problem, from equations.exact_equations."""
        logger.info(f'Multiplying {len(problem.pairs) + 1} pairs of polynomials')
        equations = exact_equations(problem, self.graph.num_edges, chunk_edges, use_numpy)
        logger.info(f'Found {len(equations)} distinct equations')
        return equations

    def get_numeric_equations(
        self, problem: Problem, extra_points: int = 10, seed: int = 0, use_numpy: bool = True
    ) -> list[Equation]:
        """
        A linear system of a Problem from random evaluations, from
        evaluation.numeric_equations. Use verify_solution to confirm a
        solution symbolically.
        """
        logger.info(f'Evaluating polynomials at {len(problem.pairs) + extra_points} random points')
        return numeric_equations(problem, self.graph.num_edges, extra_points, seed, use_numpy)

    def verify_solution(self, solution: Solution) -> bool:
        """
        Confirms symbolically that every member of a solution family returned
        by solve_problem satisfies the identity, using the exact system of
        get_equations.
        """
        # The numbers of parts of the P_i and Q_j, from any pair of patterns
        p, q = next(iter(solution.coefficients))
        P, Q = solution.product
        problem = self._problem(P, Q, len(set(p)), len(set(q)))
        return self._verify(problem, solution)

    def _verify(self, problem: Problem, solution: Solution) -> bool:
        logger.info('Verifying solution symbolically')
        coefficients = [solution.coefficients[pair] for pair in problem.pairs]
        for row, constant in self.get_equations(problem):
            lhs = sum((r * c for r, c in zip(row, coefficients) if r), sym.Integer(0))
            if sym.expand(lhs - constant) != 0:
                return False
        return True

    def solve_system(
        self, equations: Collection[Equation], num_vars: int, modular: bool = True
    ) -> tuple[sym.Expr, ...] | None:
        """
        The general solution of a linear system of (row, constant) equations,
        as a tuple of sympy expressions in free parameters x_0, x_1, ..., or
        None if there is no solution. Solved modulo primes (see modular.py),
        or with modular=False over the rationals by sympy.linsolve, which
        gives the same result but is much slower.
        """
        logger.info('Solving linear system')
        if modular:
            return solve_modular(equations, num_vars)

        xs = sym.symbols(f'x_:{num_vars}')
        # Sort to keep the parametrization of the solution reproducible
        equations = sorted(equations)
        A = sym.Matrix([row for row, _ in equations])
        b = sym.Matrix([constant for _, constant in equations])
        solutions = sym.linsolve((A, b), xs)
        # linsolve returns either EmptySet or a FiniteSet with one parametric solution
        return cast(tuple[sym.Expr, ...], solutions.args[0].args) if solutions.args else None

    def solve_problem(
        self,
        P: Pattern,
        Q: Pattern,
        balanced: bool = False,
        verify: bool = False,
        modular: bool = True,
    ) -> Solution | None:
        """
        Finds the family of identities phi_P * phi_Q = sum of c * phi_P_i * phi_Q_j
        from random evaluations, as a Solution, or None if there is none.
        With verify=True the family is also confirmed symbolically, which is
        much slower, and the result stored in Solution.verified. See
        solve_system for modular.
        """
        problem = self.problem(P, Q, balanced)
        logger.info(f'Solving for {len(problem.pairs)} unknowns')

        equations = self.get_numeric_equations(problem)
        values = self.solve_system(equations, len(problem.pairs), modular)
        if values is None:
            return None

        solution = Solution((P, Q), dict(zip(problem.pairs, values)))
        if verify:
            # Can only fail if the random points were degenerate
            solution.verified = self._verify(problem, solution)
        return solution
