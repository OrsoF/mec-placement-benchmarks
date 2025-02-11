from utils.build_instance_tree import make_tree_instance
from utils.data_management import solve
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
problem_instance, solver = solve(instance_name, solver_name)
print(solver.score)
