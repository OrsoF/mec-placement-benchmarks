"""Mobile edge computing service placement benchmarks."""

__version__ = "0.1.0"

from mecbench.api import SolveResult, SolveStatus, make_star_instance, make_tree_instance, solve
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.solution import Solution
from mecbench.domain.validation import (
    SolutionValidationError,
    ValidationReport,
    Violation,
    validate,
)
from mecbench.errors import MissingSolverDependencyError, SolverExecutionError, UnknownSolverError
from mecbench.utils.solver_registry import list_solvers

__all__ = [
    "make_tree_instance",
    "make_star_instance",
    "solve",
    "validate",
    "list_solvers",
    "ProblemInstance",
    "Solution",
    "SolveResult",
    "SolveStatus",
    "ValidationReport",
    "Violation",
    "SolutionValidationError",
    "UnknownSolverError",
    "MissingSolverDependencyError",
    "SolverExecutionError",
]
