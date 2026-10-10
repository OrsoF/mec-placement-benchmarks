from mecbench.instances.synthetic import make_tree_instance
from mecbench.benchmarks.io import solve
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_size = 10

# Small instance
users = 4
neighbors = 1

# Small instance
users = 10
neighbors = 2

# # Average instance
users = 20
neighbors = 4

# # Large instance
# users = 30
# neighbors = 6

instance_name = "tree_{}_{}".format(users, neighbors)
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

solver_name = "ILP_arc"
solver_name = "ILP_path"
solver_name = "heuristic"
solver_name = "cma"


problem_instance, solver = solve(instance_name, solver_name, verbose=True)
print(solver.score)
