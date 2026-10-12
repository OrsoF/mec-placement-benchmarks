import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mecbench.instances.synthetic import make_star_instance
from mecbench.compat.experiment_io import solve
from mecbench.compat.pickle_io import pickle_load, pickle_save
from mecbench.solvers.base import GenericSolver

from mecbench.solvers.exact.ilp_arc import Solver as Solver_ILP_arc
from mecbench.solvers.exact.ilp_path import Solver as Solver_ILP_path
from mecbench.solvers.heuristics.heuristic import Solver as Heuristic
from mecbench.solvers.evolutionary.cma import Solver as Cma
from mecbench.solvers.rl.a2c import Solver as A2c
from mecbench.solvers.rl.ppo import Solver as Ppo
from mecbench.solvers.exact.ilp_arc_95_latency import Solver as Solver_ILP_arc_95
from mecbench.solvers.exact.ilp_arc_heuristic_warmstart import Solver as Solver_ILP_arc_hws

CSV_PATH = "results_03_02_25.csv"
mecbench.solvers = {
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

    problem_instance = make_star_instance(
        users_per_group=user_count,
        neighbors_connection=neighbors,
        number_of_groups=groups,
    )

    for solver_name, solver in mecbench.solvers.items():
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


for user_count in list(range(4, 20, 2)) + list(range(20, 130, 5)):
    experience(user_count)
