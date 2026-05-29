from instances.builder import make_star_instance

instance = make_star_instance(5, 2, 3)

from solvers.rl.a2c import Solver

solver = Solver(instance, None, 100, 5000, 1000)
