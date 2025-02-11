from problem_classes.probleminstance import ProblemInstance
from problem_classes.solvinginstance import SolvingInstance
import numpy as np
import time


class GenericSolver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.solver_name = "generic_solver"

        self.x_variable: np.ndarray
        self.y_variable: np.ndarray
        self.q_variable: np.ndarray

        self.score: float
        self.runtime: float
        self.build_time: float = 0

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
            self.x_variable is not None
            and self.y_variable is not None
            and self.q_variable is not None
        ), "Result variables not set."
        self.runtime = time.time() - self.start_time
