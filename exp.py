"""Run registered solvers on tree or star SPP instances."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from instances.builder import create_star_problem_instance, create_tree_problem_instance


INSTANCES = ("tree", "star")
SOLVERS = ("ilp_arc", "heuristic", "fast_heuristic", "cma", "PPO", "HRL")
RESULT_COLUMNS = (
    "instance",
    "solver",
    "users",
    "neighbors",
    "groups",
    "score",
    "runtime",
    "buildtime",
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run SPP benchmark experiments.")
    parser.add_argument("--instance", choices=INSTANCES, default="tree")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--start-users", type=int)
    parser.add_argument("--stop-users", type=int)
    parser.add_argument("--step-users", type=int)
    parser.add_argument("--ilp-time-limit", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def set_instance_defaults(args):
    if args.start_users is None:
        args.start_users = 20 if args.instance == "tree" else 15
    if args.stop_users is None:
        args.stop_users = 410 if args.instance == "tree" else 125
    if args.step_users is None:
        args.step_users = 10 if args.instance == "tree" else 5
    if args.output is None:
        args.output = Path("artifacts") / "data" / f"exp_{args.instance}.csv"
    if args.figures_dir is None:
        args.figures_dir = Path("artifacts") / "figures" / args.instance
    return args


def make_solver(name: str, problem):
    if name == "ilp_arc":
        from solvers.exact.ilp_arc import Solver
    elif name == "heuristic":
        from solvers.heuristic.heuristic import Solver
    elif name == "fast_heuristic":
        from solvers.heuristic.fast_heuristic import Solver
    elif name == "cma":
        from solvers.evolutionary.cma import Solver
    elif name == "PPO":
        from solvers.rl.ppo import Solver
    elif name == "HRL":
        from solvers.rl.hrl import Solver
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


def result_row(args, name: str, users: int, groups: int, solver) -> dict:
    return {
        "instance": args.instance,
        "solver": name,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": solver.score,
        "runtime": solver.runtime,
        "buildtime": getattr(solver, "build_time", 0.0) or 0.0,
    }


def failed_row(args, name: str, users: int, groups: int, runtime: float) -> dict:
    return {
        "instance": args.instance,
        "solver": name,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": "",
        "runtime": runtime,
        "buildtime": "",
    }


def run_solver(name: str, problem, ilp_time_limit: float):
    solver = make_solver(name, problem)
    if name == "ilp_arc":
        solver.run(options={"TimeLimit": ilp_time_limit})
    else:
        solver.run()
    return solver


def plot_results(args):
    df = pd.read_csv(args.output)
    mask = (df["instance"] == args.instance) & (df["neighbors"] == args.neighbors)
    if args.instance == "star":
        mask &= df["groups"] == args.groups
    df = df[mask].copy()
    if df.empty:
        return

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df.groupby(["solver", "users"], as_index=False)[["score", "runtime"]].mean()
    df = df.sort_values(["solver", "users"])

    for metric, ylabel, filename in (
        ("score", "objective", f"{args.instance}_objective_vs_graph_size.pdf"),
        ("runtime", "solving time (s)", f"{args.instance}_runtime_vs_graph_size.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for solver, g in df.dropna(subset=[metric]).groupby("solver"):
            ax.plot(g["users"], g[metric], marker="o", linewidth=1.6, label=solver)

        ax.set_xlabel("graph size (number of users)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.figures_dir / filename, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None):
    args = set_instance_defaults(parse_args(argv))
    ensure_csv(args.output)

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
                runtime = time.time() - start
                row = failed_row(args, name, users, groups, runtime)
                msg = f"failed after {runtime:.3f}s: {type(exc).__name__}: {exc}"

            append_row(args.output, row)
            print(f"{args.instance} users={users} solver={name} {msg}")

    if not args.no_plots:
        plot_results(args)


if __name__ == "__main__":
    main()
