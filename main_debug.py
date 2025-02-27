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
from solvers.ILP_arc_heuristic_warmstart import Solver as Solver_ILP_arc_hws
from solvers.ILP_arc_heuristic_placement import (
    Solver as Solver_ILP_arc_heuristic_placement,
)
from solvers.ILP_arc_rl_warmstart import Solver as Solver_ILP_arc_rl_warm_start
from solvers.rl_PPO_options import Solver as Ppo_options

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
argparser = argparse.ArgumentParser(description="Run the tree instance")
argparser.add_argument("--instance", type=str, required=True)
argparser.add_argument("--user", type=int, required=True)
args = argparser.parse_args()
instance = args.instance
assert instance in {"tree", "parallel"}

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = f"results_{current_date}.csv"
# CSV_PATH = "results_12_02_25.csv"
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(
        columns=[
            "instance",
            "solver",
            "users",
            "neighbors",
            "groups",
            "score",
            "runtime",
            "buildtime",
        ]
    )
    df.to_csv(CSV_PATH, index=False)

solvers = {
    "ilp_arc": Solver_ILP_arc,
    "ilp_arc_95": Solver_ILP_arc_95,
    "ilp_arc_heuristic_placement": Solver_ILP_arc_heuristic_placement,
    "ilp_arc_heuristic_warm_start": Solver_ILP_arc_hws,
    "heuristic": Heuristic,
    "ilp_arc_rl_warm_start": Solver_ILP_arc_rl_warm_start,
    "PPO": Ppo,
    # "ilp_path": Solver_ILP_path,
    # "PPO_options": Ppo_options,
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

    total_user_count = users if instance == "tree" else users * groups

    for solver_name, solver in solvers.items():

        solver: GenericSolver = solver(problem_instance)

        if "ilp_arc" in solver_name and problem_instance.number_of_service > 150:
            continue

        solver.run()

        if not hasattr(solver, "build_time"):
            solver.build_time = 0.0

        print(
            f"Solver {solver_name} with size {total_user_count} : score {solver.score} and runtime {solver.runtime}"
        )

        update_csv(
            instance,
            solver_name,
            total_user_count,
            neighbors,
            groups,
            solver.score,
            solver.runtime,
            solver.build_time,
        )


experience(args.user)
