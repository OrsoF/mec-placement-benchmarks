from problem_classes.probleminstance import ProblemInstance
import numpy as np


class GenericSolver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.runtime: float
        self.lower_bound: float
        self.upper_bound: float
        self.solver_name = ""

        self.x_variable: np.ndarray
        self.y_variable: np.ndarray
        self.q_variable: np.ndarray

    def run(
        self,
        initial_variable: tuple,
    ):
        # To be implemented in the real solver.
        pass
