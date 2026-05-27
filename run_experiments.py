"""Run benchmark experiments and append solver results to a dated CSV file."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd

from core.generic_solver import GenericSolver
from core.problem_classes.problem_instance import ProblemInstance
from instances.large_builder import make_large_instance
from instances.parallel_builder import build_star_instance
from instances.tree_builder import make_tree_instance
from solvers.exact.ilp_arc import Solver as IlpArcSolver
from solvers.heuristic.heuristic import Solver as HeuristicSolver
from solvers.rl.ppo import Solver as PpoSolver


INSTANCE_TYPES = ("tree", "parallel", "large")

RESULT_COLUMNS = [
    "instance",
    "solver",
    "users",
    "neighbors",
    "groups",
    "score",
    "runtime",
    "buildtime",
]

SOLVER_REGISTRY: dict[str, Callable[[ProblemInstance], GenericSolver]] = {
    "ilp_arc": IlpArcSolver,
    "heuristic": HeuristicSolver,
    "PPO": PpoSolver,
}

DEFAULT_SOLVERS = ["ilp_arc", "heuristic", "PPO"]


@dataclass(frozen=True)
class ExperimentConfig:
    instance_type: str
    result_path: Path
    solver_names: list[str]
    neighbors: int
    groups: int
    user_counts: range


def default_user_counts(instance_type: str) -> range:
    if instance_type == "tree":
        return range(140, 410, 10)
    return range(35, 125, 5)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run SPP benchmark experiments.")
    parser.add_argument("--instance", choices=["tree", "parallel"], required=True)
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=sorted(SOLVER_REGISTRY),
        default=DEFAULT_SOLVERS,
        help="Solvers to run, in order.",
    )
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--start-users", type=int)
    parser.add_argument("--stop-users", type=int)
    parser.add_argument("--step-users", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_result_path(),
        help="CSV file where results are appended.",
    )
    args = parser.parse_args()

    user_counts = resolve_user_counts(
        args.instance,
        args.start_users,
        args.stop_users,
        args.step_users,
    )
    return ExperimentConfig(
        instance_type=args.instance,
        result_path=args.output,
        solver_names=args.solvers,
        neighbors=args.neighbors,
        groups=args.groups,
        user_counts=user_counts,
    )


def default_result_path() -> Path:
    current_date = datetime.now().strftime("%d_%m_%y")
    return Path("artifacts") / "data" / f"results_{current_date}.csv"


def resolve_user_counts(
    instance_type: str,
    start: int | None,
    stop: int | None,
    step: int | None,
) -> range:
    default_range = default_user_counts(instance_type)
    return range(
        default_range.start if start is None else start,
        default_range.stop if stop is None else stop,
        default_range.step if step is None else step,
    )


def ensure_results_file(result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(result_path, index=False)


def build_problem_instance(
    instance_type: str,
    users: int,
    neighbors: int,
    groups: int,
    gml_path: Path | str | None = None,
) -> ProblemInstance:
    if instance_type == "tree":
        return make_tree_instance(users, neighbors)
    if instance_type == "parallel":
        return build_star_instance(groups, users, neighbors)
    assert gml_path is not None, "large instances require a GML topology path."
    return make_large_instance(gml_path, n_services=users)


def total_user_count(instance_type: str, users: int, groups: int) -> int:
    if instance_type in {"tree", "large"}:
        return users
    return users * groups


def should_skip_solver(solver_name: str, problem_instance: ProblemInstance) -> bool:
    return "ilp_arc" in solver_name and problem_instance.number_of_service > 150


def append_result(
    result_path: Path,
    instance_type: str,
    solver_name: str,
    users: int,
    neighbors: int,
    groups: int,
    solver: GenericSolver,
) -> None:
    append_result_row(
        result_path,
        solver_result_row(
            instance_type,
            solver_name,
            users,
            neighbors,
            groups,
            solver,
        ),
    )


def run_solver(
    solver_factory: Callable[[ProblemInstance], GenericSolver],
    problem_instance: ProblemInstance,
) -> GenericSolver:
    solver = solver_factory(problem_instance)
    solver.run()
    return solver


def solver_result_row(
    instance_type: str,
    solver_name: str,
    users: int,
    neighbors: int,
    groups: int,
    solver: GenericSolver,
) -> dict[str, object]:
    return {
        "instance": instance_type,
        "solver": solver_name,
        "users": users,
        "neighbors": neighbors,
        "groups": groups,
        "score": solver.score,
        "runtime": solver.runtime,
        "buildtime": solver.build_time or 0.0,
    }


def append_result_row(result_path: Path, row: dict[str, object]) -> None:
    ensure_results_file(result_path)
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        result_path,
        mode="a",
        header=False,
        index=False,
    )


def experience_all_solvers(
    problem_instance: ProblemInstance,
    solver_names: list[str] | None = None,
    instance_type: str = "custom",
    users: int | None = None,
    neighbors: int | None = None,
    groups: int | None = None,
    result_path: Path | None = None,
) -> list[dict[str, object]]:
    """Run every selected solver on one already-built instance."""
    rows = []
    solver_names = DEFAULT_SOLVERS if solver_names is None else solver_names
    users = problem_instance.number_of_service if users is None else users
    neighbors = 0 if neighbors is None else neighbors
    groups = 1 if groups is None else groups

    for solver_name in solver_names:
        if should_skip_solver(solver_name, problem_instance):
            continue
        solver = run_solver(SOLVER_REGISTRY[solver_name], problem_instance)
        row = solver_result_row(
            instance_type,
            solver_name,
            users,
            neighbors,
            groups,
            solver,
        )
        rows.append(row)
        if result_path is not None:
            append_result_row(result_path, row)
    return rows


def experience_all_instance_types(
    solver_name: str,
    users: int,
    neighbors: int = 1,
    groups: int = 4,
    instance_types: tuple[str, ...] = INSTANCE_TYPES,
    gml_path: Path | str | None = None,
    result_path: Path | None = None,
) -> list[dict[str, object]]:
    """Run one solver on one size from every selected instance family."""
    assert solver_name in SOLVER_REGISTRY
    rows = []
    for instance_type in instance_types:
        if instance_type == "large" and gml_path is None:
            continue
        instance = build_problem_instance(
            instance_type,
            users,
            neighbors,
            groups,
            gml_path=gml_path,
        )
        if should_skip_solver(solver_name, instance):
            continue
        solver = run_solver(SOLVER_REGISTRY[solver_name], instance)
        row = solver_result_row(
            instance_type,
            solver_name,
            total_user_count(instance_type, users, groups),
            neighbors,
            groups,
            solver,
        )
        rows.append(row)
        if result_path is not None:
            append_result_row(result_path, row)
    return rows


def run_one_size(config: ExperimentConfig, users: int) -> None:
    problem_instance = build_problem_instance(
        config.instance_type,
        users,
        config.neighbors,
        config.groups,
    )
    user_count = total_user_count(config.instance_type, users, config.groups)

    for solver_name in config.solver_names:
        if should_skip_solver(solver_name, problem_instance):
            print(f"Skipping {solver_name} for size {user_count}: instance too large.")
            continue

        solver = run_solver(
            SOLVER_REGISTRY[solver_name],
            problem_instance,
        )
        print_result(solver_name, user_count, solver)
        append_result(
            config.result_path,
            config.instance_type,
            solver_name,
            user_count,
            config.neighbors,
            config.groups,
            solver,
        )


def print_result(solver_name: str, users: int, solver: GenericSolver) -> None:
    print(
        f"Solver {solver_name} with size {users}: "
        f"score {solver.score} and runtime {solver.runtime}"
    )


def run_experiments(config: ExperimentConfig) -> None:
    ensure_results_file(config.result_path)
    for users in config.user_counts:
        run_one_size(config, users)


def main() -> None:
    run_experiments(parse_args())


if __name__ == "__main__":
    main()
