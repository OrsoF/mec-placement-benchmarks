from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os
from tqdm import trange, tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "rl_base"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
env = SPPGym(problem_instance, 10, 1000)
env.reset()

for _ in trange(20):
    action = env.action_space.sample()
    print(action)
    env.step(action)
    # env.render()