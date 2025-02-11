from utils.gym_instance import SPPEnv
from utils.build_instance_tree import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 5
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

env = SPPEnv(main_problem_instance=problem_instance)
env.reset()
max_value = 0


def objective(value: np.ndarray):
    global max_value
    S = problem_instance.number_of_service
    D = problem_instance.total_number_of_demands
    assert value.shape[0] == S + 2 * D
    env.state = value
    env.clip_state()
    try:
        env._test_solution()
        obj = np.sum(env.state[-D:])
        if obj > max_value:
            max_value = obj

        return -obj
    except AssertionError:
        return 0.0


env.reset()
x0 = np.array(env.state).astype(np.float64)
print(objective(x0))

import cma

es = cma.CMAEvolutionStrategy(x0, 2)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [objective(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()

print("Meilleure solution CMA: {}".format(max_value))

instance, solver = solve(instance_name, "ILP_arc")
print("Meilleure solution ILP-arc: {}".format(solver.score))
