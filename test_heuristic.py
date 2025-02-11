from utils.build_instance_parallel import make_parallel_instance
from solvers.heuristic import Solver
import numpy as np

instance = make_parallel_instance(5, 2, 3)
solver = Solver(instance, 3, 10)
solver.run()

print(np.round(solver.runtime, 2), solver.score)
