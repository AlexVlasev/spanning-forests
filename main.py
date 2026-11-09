"""
Searches for quadratic spanning forest polynomial identities from the command line.

Examples:

    python main.py 111 123 --graph K5 --balanced
    python main.py 1111 1234 --graph K6 --balanced --verify
    python main.py 11122 12345 --graph K6
    python main.py 111 123 --graph 0-1,1-2,2-3,3-0,0-2 --marked 0,1,3
"""

import argparse
import logging
import sys
from time import time

from graph import Graph, complete_graph
from identities import IdentitySolver


def parse_graph(spec):
    """An adjacency matrix from K<n> for the complete graph, or edges like 0-1,1-2,2-0."""
    if spec[:1] in 'Kk' and spec[1:].isdigit():
        return complete_graph(int(spec[1:]))
    try:
        edges = [tuple(map(int, edge.split('-'))) for edge in spec.split(',')]
        assert all(len(edge) == 2 for edge in edges)
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(f'expected K<n> or edges like 0-1,1-2, got {spec!r}')

    n = max(max(edge) for edge in edges) + 1
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i, j in edges:
        matrix[i][j] = matrix[j][i] = 1
    return matrix


def parse_vertices(spec):
    try:
        return [int(v) for v in spec.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected vertices like 0,1,2, got {spec!r}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Search for identities '
        'phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j '
        'between spanning forest polynomials.',
        epilog='Partitions are patterns with one digit per marked vertex, e.g. 1213.',
    )
    parser.add_argument('P', help='partition of the marked vertices in the first factor')
    parser.add_argument('Q', help='partition of the marked vertices in the second factor')
    parser.add_argument(
        '-g',
        '--graph',
        type=parse_graph,
        default='K5',
        help='K<n> or edges like 0-1,1-2 (default: K5)',
    )
    parser.add_argument(
        '-m',
        '--marked',
        type=parse_vertices,
        help='marked vertices like 0,1,2 (default: the first len(P))',
    )
    parser.add_argument(
        '--balanced', action='store_true', help='balance the number of parts of the P_i and Q_j'
    )
    parser.add_argument(
        '--verify', action='store_true', help='confirm the solution symbolically (slow)'
    )
    parser.add_argument(
        '--exact-solver',
        action='store_true',
        help='solve over the rationals with sympy instead of modulo primes (slow)',
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='hide progress messages')
    args = parser.parse_args(argv)

    marked = args.marked if args.marked is not None else list(range(len(args.P)))
    if not len(args.P) == len(args.Q) == len(marked):
        parser.error('P, Q and the marked vertices must have the same length')
    if max(marked) >= len(args.graph):
        parser.error(f'the graph has no vertex {max(marked)}')

    # Progress messages are logged at INFO level, to stderr
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(format='%(message)s', level=level, force=True)

    start = time()
    solver = IdentitySolver(Graph(args.graph), marked)
    solution = solver.solve_problem(
        args.P,
        args.Q,
        balanced=args.balanced,
        verify=args.verify,
        modular=not args.exact_solver,
    )

    print('\n====================================\n')
    if solution is None:
        print('No Solution')
    else:
        print(solution.report())
        print('\n====================================\n')
        print(solution.identity())
        if solution.verified is True:
            print('\nVerified symbolically')
        elif solution.verified is False:
            print('\nSymbolic verification FAILED')

    print(f'\nComputation took: {time() - start:.2f} sec')
    return 0 if solution is not None and solution.verified is not False else 1


if __name__ == '__main__':
    sys.exit(main())
