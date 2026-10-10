"""Run registered solvers and persist benchmark results."""
import csv
import time
from pathlib import Path
from mecbench.instances.synthetic import create_star_problem_instance, create_tree_problem_instance
from mecbench.benchmarks.metrics import RESULT_COLUMNS, result_row, failed_row
from mecbench.domain.validation import validate_arc_solution

INSTANCES = ("tree", "star")
SOLVERS = ("ilp_arc", "heuristic", "fast_heuristic", "cma", "PPO", "HRL")

def make_solver(name: str, problem):
    if name == "ilp_arc":
        from mecbench.solvers.exact.ilp_arc import Solver
    elif name == "heuristic":
        from mecbench.solvers.heuristics.heuristic import Solver
    elif name == "fast_heuristic":
        from mecbench.solvers.heuristics.fast_heuristic import Solver
    elif name == "cma":
        from mecbench.solvers.evolutionary.cma import Solver
    elif name == "PPO":
        from mecbench.solvers.rl.ppo import Solver
    elif name == "HRL":
        from mecbench.solvers.rl.hrl import Solver
    else:
        raise ValueError(name)

    return Solver(problem)



def ensure_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writeheader()



def append_row(path: Path, row: dict):
    ensure_csv(path)
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writerow(row)



def should_skip(name: str, problem) -> bool:
    return name == "ilp_arc" and problem.number_of_service > 150



def make_problem(args, size: int):
    if args.instance == "tree":
        return create_tree_problem_instance(size, args.neighbors), size, 1

    problem = create_star_problem_instance(args.groups, size, args.neighbors)
    return problem, args.groups * size, args.groups



def run_solver(name: str, problem, ilp_time_limit: float):
    solver = make_solver(name, problem)
    if name == "ilp_arc":
        solver.run(options={"TimeLimit": ilp_time_limit})
    else:
        solver.run()
    validate_arc_solution(problem, solver.x_var, solver.y_var, solver.q_var)
    return solver



def run_benchmarks(args):
    ensure_csv(args.output)
    failures = 0

    for size in range(args.start_users, args.stop_users, args.step_users):
        problem, users, groups = make_problem(args, size)

        for name in args.solvers:
            if should_skip(name, problem):
                print(f"Skipping {name} for {args.instance} size {users}: instance too large.")
                continue

            start = time.time()
            try:
                solver = run_solver(name, problem, args.ilp_time_limit)
                row = result_row(args, name, users, groups, solver)
                msg = f"score={solver.score} runtime={solver.runtime}"
            except Exception as exc:
                failures += 1
                runtime = time.time() - start
                row = failed_row(args, name, users, groups, runtime)
                msg = f"failed after {runtime:.3f}s: {type(exc).__name__}: {exc}"

            append_row(args.output, row)
            print(f"{args.instance} users={users} solver={name} {msg}")

    if not args.no_plots:
        from mecbench.benchmarks.reporting import plot_results
        plot_results(args)
    return 1 if failures else 0
