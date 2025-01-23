import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from utils.pickle import pickle_load, pickle_save
from utils.generic_solver import GenericSolver

from solvers.ILP_arc import Solver as Solver_ILP_arc
from solvers.ILP_path import Solver as Solver_ILP_path
from solvers.heuristic import Solver as Heuristic
from solvers.cma import Solver as Cma
from solvers.rl_A2C import Solver as A2c
from solvers.rl_PPO import Solver as Ppo

CSV_PATH = "results.csv"
solvers = {
    "ilp_arc": Solver_ILP_arc,
    # "ilp_path": Solver_ILP_path,
    "heuristic": Heuristic,
    "cma": Cma,
    "A2C": A2c,
    "PPO": Ppo,
}


def update_csv(
    instance: str,
    solver: str,
    users: str,
    neighbors: str,
    groups: str,
    score: float,
    runtime: float,
    buildtime: float,
) -> None:
    df = pd.read_csv(CSV_PATH)
    new_df = pd.DataFrame(
        {
            "instance": instance,
            "solver": solver,
            "users": users,
            "neighbors": neighbors,
            "groups": groups,
            "score": score,
            "runtime": runtime,
            "buildtime": buildtime,
        },
        index=[len(df)],
    )
    df = pd.concat([df, new_df], axis=0)
    df.to_csv(CSV_PATH, index=False)


def experience(users: int):
    neighbors, groups, instance = 1, 1, "tree"

    problem_instance = make_tree_instance(users, neighbors)

    for solver_name, solver in solvers.items():
        print(solver_name)
        solver: GenericSolver = solver(problem_instance)
        solver.run()

        if not hasattr(solver, "build_time"):
            solver.build_time = 0.0

        update_csv(
            instance,
            solver_name,
            users,
            neighbors,
            groups,
            solver.score,
            solver.runtime,
            solver.build_time,
        )


for user_count in range(4, 100, 6):
    experience(user_count)
