from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 1
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "rl_fast"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

from utils.SPPGym_fast import SPPGymFast

env = SPPGymFast(problem_instance, 10, 2000)
env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    env.step(action)
