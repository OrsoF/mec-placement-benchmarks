"""One lazy registry for the API, CLI and historical benchmark helpers."""
from dataclasses import dataclass
import importlib
import inspect

from mecbench.errors import MissingSolverDependencyError, UnknownSolverError


@dataclass(frozen=True)
class SolverSpec:
    module: str
    extra: str | None = None
    mathematical: bool = False


SOLVER_MODULES = {
    "ilp_arc": "mecbench.solvers.exact.ilp_arc",
    "ilp_arc_95_latency": "mecbench.solvers.exact.ilp_arc_95_latency",
    "ilp_path": "mecbench.solvers.exact.ilp_path",
    "milp_arc": "mecbench.solvers.exact.milp_arc",
    "milp_path": "mecbench.solvers.exact.milp_path",
    "milp_arc_full": "mecbench.solvers.exact.milp_arc_full",
    "ilp_arc_heuristic_placement": "mecbench.solvers.exact.ilp_arc_heuristic_placement",
    "ilp_arc_heuristic_warmstart": "mecbench.solvers.exact.ilp_arc_heuristic_warmstart",
    "ilp_arc_rl_warmstart": "mecbench.solvers.exact.ilp_arc_rl_warmstart",
    "lp_arc": "mecbench.solvers.relaxations.lp_arc",
    "lp_path": "mecbench.solvers.relaxations.lp_path",
    "lp_path_latency_tightened": "mecbench.solvers.relaxations.lp_path_latency_tightened",
    "a2c": "mecbench.solvers.rl.a2c",
    "ppo": "mecbench.solvers.rl.ppo",
    "ppo_options": "mecbench.solvers.rl.ppo_options",
    "hrl": "mecbench.solvers.rl.hrl",
    "heuristic": "mecbench.solvers.heuristics.heuristic",
    "heuristicprecise": "mecbench.solvers.heuristics.heuristicprecise",
    "fast_heuristic": "mecbench.solvers.heuristics.fast_heuristic",
    "discrete": "mecbench.solvers.evolutionary.discrete",
    "cma": "mecbench.solvers.evolutionary.cma",
}
SOLVER_FILE_ALIASES = {
    "ILP_arc": "ilp_arc", "ILP_arc_95": "ilp_arc_95_latency", "ILP_path": "ilp_path",
    "MILP_arc": "milp_arc", "MILP_path": "milp_path", "milp_arc_exact": "milp_arc_full",
    "ILP_arc_heuristic_placement": "ilp_arc_heuristic_placement",
    "ILP_arc_heuristic_warmstart": "ilp_arc_heuristic_warmstart",
    "ILP_arc_rl_warmstart": "ilp_arc_rl_warmstart",
    "LP_arc": "lp_arc", "LP_path": "lp_path", "LP_path_strong_latency": "lp_path_latency_tightened",
    "rl_A2C": "a2c", "rl_PPO": "ppo", "rl_PPO_options": "ppo_options",
    "base_heuristic": "heuristic", "rl_base": "a2c", "PPO": "ppo", "HRL": "hrl",
}


def _spec(module):
    mathematical = ".exact." in module or ".relaxations." in module
    extra = "exact,rl" if module.endswith("ilp_arc_rl_warmstart") else (
        "exact" if mathematical else "rl" if ".rl." in module else
        "evolutionary" if module.endswith(".cma") else None
    )
    return SolverSpec(module, extra, mathematical)


REGISTRY = {name: _spec(module) for name, module in SOLVER_MODULES.items()}


def list_solvers() -> tuple[str, ...]:
    """List canonical names without importing optional dependencies."""
    return tuple(REGISTRY)


def canonical_name(name: str) -> str:
    if not isinstance(name, str):
        raise UnknownSolverError("solver must be a registered name; use list_solvers().")
    name = SOLVER_FILE_ALIASES.get(name, name)
    if name not in REGISTRY:
        raise UnknownSolverError(f"Unknown solver {name!r}. Available: {', '.join(list_solvers())}.")
    return name


def dependency_error(name, error):
    extra = REGISTRY[name].extra
    install = f"mec-placement-benchmarks[{extra}]" if extra else "mec-placement-benchmarks"
    return MissingSolverDependencyError(
        f"Solver {name!r} needs {error.name!r}. Install with: python -m pip install '{install}'."
    )


def make_solver(name, problem, *, seed=0, **options):
    name = canonical_name(name)
    try:
        cls = importlib.import_module(REGISTRY[name].module).Solver
        parameters = inspect.signature(cls).parameters
        if "seed" in parameters:
            options = {**options, "seed": seed}
        inspect.signature(cls).bind(problem, **options)
        return cls(problem, **options)
    except ModuleNotFoundError as error:
        raise dependency_error(name, error) from error
