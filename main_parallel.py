import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.make_parallel_instance import make_parallel_tree_instance
from utils.data_management import solve
from utils.pickle import pickle_load, pickle_save
from utils.generic_solver import GenericSolver

from solvers.ILP_arc import Solver as Solver_ILP_arc
from solvers.ILP_path import Solver as Solver_ILP_path
from solvers.heuristic import Solver as Heuristic
from solvers.cma import Solver as Cma
from solvers.rl_A2C import Solver as A2c
from solvers.rl_PPO import Solver as Ppo
from solvers.ILP_arc_95 import Solver as Solver_ILP_arc_95
from solvers.ILP_arc_hws import Solver as Solver_ILP_arc_hws

CSV_PATH = "results_03_02_25.csv"
solvers = {
    # "ilp_arc": Solver_ILP_arc,
    # "ilp_path": Solver_ILP_path,
    # "heuristic": Heuristic,
    # "cma": Cma,
    # "A2C": A2c,
    # "PPO": Ppo,
    # "ilp_arc_95": Solver_ILP_arc_95,
    "ilp_arc_hws": Solver_ILP_arc_hws,
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
    neighbors, groups, instance = 1, 4, "parallel"

    problem_instance = make_parallel_tree_instance(
        number_of_users_per_group=user_count,
        neighbors_connection=neighbors,
        number_of_groups=groups,
    )

    for solver_name, solver in solvers.items():
        print(solver_name)
        solver: GenericSolver = solver(problem_instance)
        try:
            solver.run()
        except:
            print(f"Solver {solver_name} with size {users} : failed")
            continue

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


for user_count in [
    # 4,
    # 6,
    # 8,
    # 10,
    # 12,
    # 14,
    # 16,
    18,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
    105,
    110,
    115,
    120,
    125,
]:
    experience(user_count)
