from rl.gym_instance import SPPEnv
from instances.build_tree import make_tree_instance
from data_io.data_management import solve
from stable_baselines3 import A2C
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "rl_base"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
problem_instance, solver = solve(instance_name, solver_name, verbose=True)
