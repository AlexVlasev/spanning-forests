"""
Searches for quadratic spanning forest polynomial identities from the command line.

Examples:

    python main.py 111 123 --graph K5 --balanced
    python main.py 1111 1234 --graph K6 --balanced --verify
    python main.py 11122 12345 --graph K6
    python main.py 111 123 --graph 0-1,1-2,2-3,3-0,0-2 --marked 0,1,3
    python main.py --relations 2,2,3 --marked 0,1,2,3 --graph K6
    python main.py --relations 1,3 2,2 --marked 0,1,2 --graph K5 --show-relations
    python main.py --relations 1,4,4 3,3,3 --marked 0,1,2,3 --sweep 8
    python main.py --relations 3,3,3 --product 1111,1234,1234 --marked 0,1,2,3 --graph K6
"""

import argparse
import logging
import sys
from time import time

from graph import Graph, complete_graph
from identities import IdentitySolver
from relations import (
    expand_product,
    find_relations,
    format_relation,
    product_terms,
    relation_counts,
    relation_dimension,
)


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


def parse_shape(spec):
    try:
        return tuple(int(parts) for parts in spec.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected numbers of parts like 2,3, got {spec!r}')


def parse_product(spec):
    return tuple(spec.split(','))


def print_relations(graph, marked, args):
    shapes = args.relations
    print(f'Products : {len(product_terms(graph, marked, shapes))}')
    print(f'Dimension: {relation_dimension(graph, marked, shapes)}')
    if args.new:
        counts = relation_counts(graph, marked, shapes)
        print(f'Inherited: {counts.inherited}')
        print(f'New      : {counts.new}')
    if args.show_relations:
        for i, relation in enumerate(find_relations(graph, marked, shapes), 1):
            print(f'\nRelation {i}:\n{format_relation(relation)}')


def print_sweep(marked, shapes, largest):
    """The counts on each complete graph that contains the marked vertices, up to K_largest."""
    print(f'{"Graph":<6}{"Products":>10}{"Dimension":>11}{"Inherited":>11}{"New":>7}')
    for n in range(max(marked) + 1, largest + 1):
        counts = relation_counts(Graph(complete_graph(n)), marked, shapes)
        print(
            f'{"K" + str(n):<6}{counts.products:>10}{counts.dimension:>11}'
            f'{counts.inherited:>11}{counts.new:>7}',
            flush=True,
        )


def print_solution(solution):
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Search for identities '
        'phi_P * phi_Q = sum of c(P_i, Q_j) * phi_P_i * phi_Q_j '
        'between spanning forest polynomials.',
        epilog='Partitions are patterns with one digit per marked vertex, e.g. 1213.',
    )
    parser.add_argument(
        'P', nargs='?', help='partition of the marked vertices in the first factor'
    )
    parser.add_argument(
        'Q', nargs='?', help='partition of the marked vertices in the second factor'
    )
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
    parser.add_argument(
        '--relations',
        nargs='+',
        type=parse_shape,
        metavar='SHAPE',
        help='instead of P and Q, the dimension of the relations between all products '
        'with these numbers of parts, like 2,3 or 2,2,3 (needs --marked)',
    )
    parser.add_argument(
        '--show-relations',
        action='store_true',
        help='with --relations, also print a basis of the relations (slower)',
    )
    parser.add_argument(
        '--new',
        action='store_true',
        help='with --relations, also split the dimension into inherited and new relations',
    )
    parser.add_argument(
        '--sweep',
        type=int,
        metavar='N',
        help='with --relations, instead of --graph, print the counts on complete graphs up to K_N',
    )
    parser.add_argument(
        '--product',
        type=parse_product,
        metavar='PATTERNS',
        help='with --relations, expand this product, like 1111,1234,1234, in those shapes',
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='hide progress messages')
    args = parser.parse_args(argv)

    if args.relations is not None:
        if args.P is not None or args.marked is None:
            parser.error('--relations needs --marked, and takes no P or Q')
        marked = args.marked
        if args.product is not None:
            if args.show_relations or args.new or args.sweep is not None or args.verify:
                parser.error('--product takes no --show-relations, --new, --sweep or --verify')
            if any(len(pattern) != len(marked) for pattern in args.product):
                parser.error('every pattern of --product needs one digit per marked vertex')
    else:
        if args.P is None or args.Q is None:
            parser.error('expected P and Q, or --relations')
        if args.show_relations or args.new or args.sweep is not None or args.product is not None:
            parser.error('--show-relations, --new, --sweep and --product need --relations')
        marked = args.marked if args.marked is not None else list(range(len(args.P)))
        if not len(args.P) == len(args.Q) == len(marked):
            parser.error('P, Q and the marked vertices must have the same length')
    if args.sweep is not None:
        if max(marked) >= args.sweep:
            parser.error(f'K{args.sweep} has no vertex {max(marked)}')
    elif max(marked) >= len(args.graph):
        parser.error(f'the graph has no vertex {max(marked)}')

    # Progress messages are logged at INFO level, to stderr
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(format='%(message)s', level=level, force=True)

    start = time()
    if args.relations is not None:
        try:
            if args.product is not None:
                solution = expand_product(Graph(args.graph), marked, args.product, args.relations)
                print_solution(solution)
                print(f'\nComputation took: {time() - start:.2f} sec')
                return 0 if solution is not None else 1
            if args.sweep is not None:
                print_sweep(marked, args.relations, args.sweep)
            else:
                print_relations(Graph(args.graph), marked, args)
        except ValueError as error:
            parser.error(str(error))
        print(f'\nComputation took: {time() - start:.2f} sec')
        return 0

    solver = IdentitySolver(Graph(args.graph), marked)
    solution = solver.solve_problem(
        args.P,
        args.Q,
        balanced=args.balanced,
        verify=args.verify,
        modular=not args.exact_solver,
    )

    print_solution(solution)

    print(f'\nComputation took: {time() - start:.2f} sec')
    return 0 if solution is not None and solution.verified is not False else 1


if __name__ == '__main__':
    sys.exit(main())
