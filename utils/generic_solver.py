from problem_classes.probleminstance import ProblemInstance
from problem_classes.solvinginstance import SolvingInstance
import numpy as np


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
        pass

    def _end_solving(self):
        assert (
            self.x_variable is not None
            and self.y_variable is not None
            and self.q_variable is not None
        )
        self.solving_instance = SolvingInstance(self.problem_instance)
        print(self.y_variable)
        self.solving_instance.place_service_variable(self.y_variable)
        self.solving_instance.route_demand_variable(self.x_variable, self.q_variable)
        print("Score: {}".format(self.solving_instance.get_score()))
