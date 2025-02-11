import os
import pandas as pd
import argparse
from datetime import datetime
from utils.build_instance_tree import make_tree_instance
from utils.build_instance_parallel import make_parallel_instance
from utils.data_management import solve
from utils.pickle import pickle_load, pickle_save
from utils.generic_solver import GenericSolver
from solvers.ILP_arc import Solver as Solver_ILP_arc
from solvers.ILP_arc_95 import Solver as Solver_ILP_arc_95
from solvers.ILP_path import Solver as Solver_ILP_path
from solvers.heuristic import Solver as Heuristic
from solvers.cma import Solver as Cma
from solvers.rl_A2C import Solver as A2c
from solvers.rl_PPO import Solver as Ppo
from solvers.ILP_arc_hws import Solver as Solver_ILP_arc_hws
from solvers.ILP_arc_heuristic_placement import (
    Solver as Solver_ILP_arc_heuristic_placement,
)
from solvers.rl_PPO_options import Solver as Ppo_options

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
argparser = argparse.ArgumentParser(description="Run the tree instance")
argparser.add_argument("--instance", type=str, required=True)
args = argparser.parse_args()
instance = args.instance
assert instance in {"tree", "parallel"}

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = f"results_{current_date}.csv"

solvers = {
    # "ilp_arc": Solver_ILP_arc,
    # "ilp_path": Solver_ILP_path,
    # "heuristic": Heuristic,
    # "cma": Cma,
    # "A2C": A2c,
    # "PPO": Ppo,
    # 'ilp_arc_95': Solver_ILP_arc_95,
    # "ilp_arc_hws": Solver_ILP_arc_hws,
    # "ilp_arc_heuristic_placement": Solver_ILP_arc_heuristic_placement,
    "PPO_options": Ppo_options,
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
    neighbors, groups = 1, 4

    if instance == "tree":
        problem_instance = make_tree_instance(users, neighbors)
    elif instance == "parallel":
        problem_instance = make_parallel_instance(users, neighbors, groups)

    for solver_name, solver in solvers.items():

        solver: GenericSolver = solver(problem_instance)
        solver.run()

        if not hasattr(solver, "build_time"):
            solver.build_time = 0.0

        print(
            f"Solver {solver_name} with size {users} : score {solver.score} and runtime {solver.runtime}"
        )

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

if instance == "tree":
    users = range(30, 410, 10)
elif instance == "parallel":
    users = list(range(10, 130, 5))

for user_count in users:
    experience(user_count)
