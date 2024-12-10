from utils.make_parallel_tree import make_parallel_tree_instance

instance = make_parallel_tree_instance(5, 2, 3)

from solvers.heuristic import Solver

solver = Solver(instance, 3, 10)
solver.run()