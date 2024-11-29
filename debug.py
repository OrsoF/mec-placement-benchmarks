from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)


env = SPPGym(problem_instance)
env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    if terminated:
        env.reset()
    # env.render()