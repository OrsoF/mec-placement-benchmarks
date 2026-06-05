"""Run all registered solvers on tree instances of increasing size."""

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

from instances.builder import create_tree_problem_instance


SOLVERS = ("ilp_arc", "heuristic", "PPO")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run tree SPP experiments.")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--start-users", type=int, default=20)
    parser.add_argument("--stop-users", type=int, default=410)
    parser.add_argument("--step-users", type=int, default=10)
    parser.add_argument("--ilp-time-limit", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "data" / "exp_tree.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts") / "figures" / "tree",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def make_solver(name: str, problem):
    if name == "ilp_arc":
        from solvers.exact.ilp_arc import Solver
    elif name == "heuristic":
        from solvers.heuristic.heuristic import Solver
    elif name == "PPO":
        from solvers.rl.ppo import Solver
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


def result_row(name: str, users: int, neighbors: int, solver) -> dict:
    return {
        "instance": "tree",
        "solver": name,
        "users": users,
        "neighbors": neighbors,
        "groups": 1,
        "score": solver.score,
        "runtime": solver.runtime,
        "buildtime": getattr(solver, "build_time", 0.0) or 0.0,
    }


def failed_row(name: str, users: int, neighbors: int, runtime: float) -> dict:
    return {
        "instance": "tree",
        "solver": name,
        "users": users,
        "neighbors": neighbors,
        "groups": 1,
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


def plot_results(csv_path: Path, figures_dir: Path, neighbors: int):
    df = pd.read_csv(csv_path)
    df = df[(df["instance"] == "tree") & (df["neighbors"] == neighbors)].copy()
    if df.empty:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df.groupby(["solver", "users"], as_index=False)[["score", "runtime"]].mean()
    df = df.sort_values(["solver", "users"])

    for metric, ylabel, filename in (
        ("score", "objective", "tree_objective_vs_graph_size.pdf"),
        ("runtime", "solving time (s)", "tree_runtime_vs_graph_size.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for solver, g in df.dropna(subset=[metric]).groupby("solver"):
            ax.plot(g["users"], g[metric], marker="o", linewidth=1.6, label=solver)

        ax.set_xlabel("graph size (number of users)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    ensure_csv(args.output)

    for users in range(args.start_users, args.stop_users, args.step_users):
        problem = create_tree_problem_instance(users, args.neighbors)

        for name in args.solvers:
            if should_skip(name, problem):
                print(f"Skipping {name} for tree size {users}: instance too large.")
                continue

            start = time.time()
            try:
                solver = run_solver(name, problem, args.ilp_time_limit)
                row = result_row(name, users, args.neighbors, solver)
                msg = f"score={solver.score} runtime={solver.runtime}"
            except Exception as exc:
                runtime = time.time() - start
                row = failed_row(name, users, args.neighbors, runtime)
                msg = f"failed after {runtime:.3f}s: {type(exc).__name__}: {exc}"

            append_row(args.output, row)
            print(
                f"tree users={users} solver={name} "
                f"{msg}"
            )

    if not args.no_plots:
        plot_results(args.output, args.figures_dir, args.neighbors)


if __name__ == "__main__":
    main()
