import os

from utils.make_parallel_tree import make_parallel_tree_instance
from utils.data_management import solve
from utils.pickle import pickle_load, pickle_save

from solvers.ILP_arc import Solver as Solver_ILP_arc
from solvers.ILP_path import Solver as Solver_ILP_path
from solvers.heuristic import Solver as Heuristic
from solvers.cma import Solver as Cma
from solvers.rl_A2C import Solver as A2c
from solvers.rl_PPO import Solver as Ppo

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
solvers = {
    "ilp_arc": Solver_ILP_arc,
    "ilp_path": Solver_ILP_path,
    "heuristic": Heuristic,
    "cma": Cma,
    "A2C": A2c,
    "PPO": Ppo,
}

results = {}
path_file = os.path.join("results", "dict_scores.pkl")
if not os.path.exists(path_file):
    pickle_save(results, path_file)


def experience(user_count: int, neighbor_count: int = 2):
    results = pickle_load(path_file)

    problem_instance = make_parallel_tree_instance(
        number_of_users=user_count, neighbors_connection=neighbor_count
    )
    for solver_name, solver in solvers.items():
        print(solver_name)
        solver = solver(problem_instance)
        solver.run()
        results[("tree", solver_name, user_count, neighbor_count)] = (
            float(solver.score),
            float(solver.runtime),
        )

    pickle_save(results, path_file)


for i in range(14, 30, 2):
    experience(i)
print(pickle_load(path_file))
