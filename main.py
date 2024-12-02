from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Small instance
users = 10
neighbors = 4

# # Average instance
# users = 20
# neighbors = 8

# # Large instance
# users = 30
# neighbors = 12

instance_name = "tree_{}_{}".format(users, neighbors)
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
solver_name = "ILP_arc"
solver_name = "ILP_path"
# solver_name = "rl_A2C"
problem_instance, solver = solve(instance_name, solver_name, verbose=True)
print(solver.score)
