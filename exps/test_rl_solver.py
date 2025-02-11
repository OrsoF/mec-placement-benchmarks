from utils.build_instance_parallel import make_parallel_instance

instance = make_parallel_instance(5, 2, 3)

from solvers.rl_A2C import Solver

solver = Solver(instance, None, 100, 5000, 1000)