import os
import pandas as pd
import argparse
from datetime import datetime
from instances.build_tree import make_tree_instance
from instances.build_parallel import build_star_instance
from data_io.data_management import solve
from data_io.pickle import pickle_load, pickle_save
from core.generic_solver import GenericSolver
from solvers.exact.ILP_arc import Solver as Solver_ILP_arc
from solvers.exact.ILP_arc_95 import Solver as Solver_ILP_arc_95
from solvers.exact.ILP_path import Solver as Solver_ILP_path
from solvers.heuristic.heuristic import Solver as Heuristic
from solvers.evolutionary.cma import Solver as Cma
from solvers.rl.rl_A2C import Solver as A2c
from solvers.rl.rl_PPO import Solver as Ppo
from solvers.exact.ILP_arc_heuristic_warmstart import Solver as Solver_ILP_arc_hws
from solvers.exact.ILP_arc_heuristic_placement import (
    Solver as Solver_ILP_arc_heuristic_placement,
)
from solvers.exact.ILP_arc_rl_warmstart import Solver as Solver_ILP_arc_rl_warm_start
from solvers.rl.rl_PPO_options import Solver as Ppo_options

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
argparser = argparse.ArgumentParser(description="Run the tree instance")
argparser.add_argument("--instance", type=str, required=True)
args = argparser.parse_args()
instance = args.instance
assert instance in {"tree", "parallel"}

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = os.path.join("artifacts", "data", f"results_{current_date}.csv")
# CSV_PATH = "results_12_02_25.csv"
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
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
    # "ilp_arc_95": Solver_ILP_arc_95,
    # "ilp_arc_heuristic_placement": Solver_ILP_arc_heuristic_placement,
    # "ilp_arc_heuristic_warm_start": Solver_ILP_arc_hws,
    # "ilp_arc_rl_warm_start": Solver_ILP_arc_rl_warm_start,
    # "ilp_path": Solver_ILP_path,
    "heuristic": Heuristic,
    "PPO": Ppo,
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
        problem_instance = build_star_instance(
            users, neighbors, groups
        )

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


users = range(140, 410, 10) if instance == "tree" else range(35, 125, 5)

for user_count in users:
    experience(user_count)
