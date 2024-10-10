from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 15
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "heuristic"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
problem_instance, solver = solve(instance_name, solver_name)
print(solver.score)
