from utils.make_parallel_tree import make_parallel_tree_instance
from utils.data_management import solve
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_size = 10

# Small instance
users = 5
neighbors = users // 3 - 1
groups = users - 2

# Small instance
# users = 10
# neighbors = 2

# # Average instance
# users = 20
# neighbors = 4

# # Large instance
# users = 30
# neighbors = 6

instance_name = "tree_parallel_{}_{}".format(users, neighbors)
problem_instance = make_parallel_tree_instance(users, neighbors, groups)

solver_name = "ILP_arc"
solver_name = "ILP_path"
solver_name = "heuristic"
solver_name = "cma"


problem_instance, solver = solve(
    instance_name, solver_name, verbose=True, instance=problem_instance
)
print(solver.score)
