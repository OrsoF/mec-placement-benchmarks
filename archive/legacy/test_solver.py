from instances.build_parallel import build_star_instance

number_of_groups = 5
users_per_group = 10
neighbors_connection = 1


instance = build_star_instance(
    number_of_groups,
    users_per_group,
    neighbors_connection,
)

from solvers.evolutionary.cma import Solver
import numpy as np

solver = Solver(instance)
solver.run()

print("Runtime:", np.round(solver.runtime, 2))
print("Score:", solver.score)
