from utils.build_instance_parallel import build_star_instance

users_per_group = 5
neighbors_connection = 1
demand_per_group = 5

instance = build_star_instance(
    users_per_group,
    neighbors_connection,
    demand_per_group,
)

from solvers.heuristic import Solver
import numpy as np

solver = Solver(instance, 3, 10)
solver.run()

print("Runtime:", np.round(solver.runtime, 2))
print("Score:", solver.score)
