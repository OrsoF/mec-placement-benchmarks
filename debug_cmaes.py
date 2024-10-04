from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os
import numpy as np


users = 20
neighbors = 2
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

env = SPPGym(problem_instance=problem_instance)
env.reset()


def objective(value: np.ndarray):
    S = problem_instance.number_of_service
    D = problem_instance.total_number_of_demands
    assert value.shape[0] == S + 2 * D
    env.state = value
    try:
        env._test_solution()
        return np.sum(value[-D:])
    except AssertionError:
        return 0.0


env.reset()
x0 = np.array(env.state)
print(objective(x0))

from cmaes import CMA

optimizer = CMA(mean=x0, sigma=1.0)

for generation in range(50):
    solutions = []
    for _ in range(optimizer.population_size):
        x = optimizer.ask()
        value = objective(x)
        solutions.append((x, value))
        print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
    optimizer.tell(solutions)
