from time import time

from graph import Graph, complete_graph
from identities import IdentitySolver

if __name__ == '__main__':
    start = time()

    # n = 5
    # vertices = [k for k in range(n-1)]
    # solver = IdentitySolver(Graph(complete_graph(n)), vertices, verbose=True)
    # P = '1' * (n-1)
    # Q = ''.join(map(lambda x: str(x+1), vertices))
    # solver.solve_problem(P, Q)

    # solver = IdentitySolver(Graph(complete_graph(6)), [0, 1, 2, 3, 4], verbose=True)
    # solver.solve_problem('11111', '12345', reduce=True)

    # solver = IdentitySolver(Graph(complete_graph(5)), [0, 1, 2], verbose=True)
    # solver.solve_problem('111', '123', reduce=True)

    # solver = IdentitySolver(Graph(complete_graph(6)), [0, 1, 2, 3, 4], verbose=True)
    # solver.solve_problem('11122', '12345', reduce=False)

    solver = IdentitySolver(Graph(complete_graph(5)), [0, 1, 2, 3], verbose=True)
    solver.solve_problem('1111', '1234', reduce=True)

    end = time()
    print(f'\nComputation took: {int(100*(end-start))/100} sec')
