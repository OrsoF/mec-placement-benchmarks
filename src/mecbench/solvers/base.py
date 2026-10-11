import time

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray
from mecbench.domain.problem import ProblemInstance


class GenericSolver:
    REQUIRED_OUTPUT_ATTRIBUTES = (
        "x_var",
        "y_var",
        "q_var",
        "score",
        "runtime",
        "build_time",
    )

    def __init__(self, problem_instance: ProblemInstance) -> None:
        self.problem_instance = problem_instance
        self.solver_name = "generic_solver"

        self.x_var: FloatArray | None = None
        self.y_var: FloatArray | None = None
        self.q_var: FloatArray | None = None

        self.score: float | None = None
        self.runtime: float | None = None
        self.build_time: float = 0.0
        self.termination_condition: str | None = None

    def _set_output_variables(self, x_var: ArrayLike, y_var: ArrayLike, q_var: ArrayLike) -> None:
        self.x_var = np.asarray(x_var, dtype=float)
        self.y_var = np.asarray(y_var, dtype=float)
        self.q_var = np.asarray(q_var, dtype=float)

        # Legacy aliases still used by older experiment helpers.
        self.x_variable = self.x_var
        self.y_variable = self.y_var
        self.q_variable = self.q_var

    def run(self) -> object:
        raise NotImplementedError("Subclasses must implement run().")

    def _start_solving(self) -> None:
        self.start_time = time.time()

    def _end_solving(self) -> None:
        if self.x_var is None or self.y_var is None or self.q_var is None:
            raise RuntimeError("Result variables not set.")
        self.runtime = time.time() - self.start_time
