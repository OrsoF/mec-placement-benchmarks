from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.solving_instance import SolvingInstance
import numpy as np
import time
from functools import wraps


class GenericSolver:
    REQUIRED_OUTPUT_ATTRIBUTES = (
        "x_var",
        "y_var",
        "q_var",
        "score",
        "runtime",
        "build_time",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        if getattr(original_init, "_solver_contract_wrapped", False):
            return

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._ensure_output_contract()

        wrapped_init._solver_contract_wrapped = True
        cls.__init__ = wrapped_init

    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.solver_name = "generic_solver"

        self.x_var: np.ndarray | None = None
        self.y_var: np.ndarray | None = None
        self.q_var: np.ndarray | None = None

        self.score: float | None = None
        self.runtime: float | None = None
        self.build_time: float = 0

    def _ensure_output_contract(self):
        for attribute in self.REQUIRED_OUTPUT_ATTRIBUTES:
            if hasattr(self, attribute):
                continue

            if attribute == "build_time":
                setattr(self, attribute, 0.0)
            else:
                setattr(self, attribute, None)

    def run(
        self,
        initial_variable: tuple,
    ):
        # To be implemented in the real solver.
        assert False, "To be implemented."

    def _start_solving(self):
        self.start_time = time.time()

    def _end_solving(self):
        assert (
            self.x_var is not None
            and self.y_var is not None
            and self.q_var is not None
        ), "Result variables not set."
        self.runtime = time.time() - self.start_time
