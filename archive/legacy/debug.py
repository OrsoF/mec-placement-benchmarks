from mecbench.environments.spp_env import SPPEnv
from mecbench.instances.synthetic import make_star_instance
from mecbench.benchmarks.io import solve
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
problem_instance = make_star_instance(
    number_of_users=users, neighbors_connection=neighbors
)


env = SPPEnv(problem_instance, 500)
env.reset()

from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import MlpPolicy

training = A2C(MlpPolicy, env, verbose=1)
training.learn(1000, progress_bar=True)
