from utils.build_instance_parallel import build_star_instance

instance = build_star_instance(5, 2, 3)

from solvers.rl_A2C import Solver

solver = Solver(instance, None, 100, 5000, 1000)