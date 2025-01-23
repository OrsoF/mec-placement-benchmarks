from utils.make_parallel_instance import make_parallel_tree_instance

instance = make_parallel_tree_instance(5, 2, 3)

from solvers.rl_A2C import Solver

solver = Solver(instance, None, 100, 5000, 1000)