"Solveur RL PPO"

from core.generic_solver import GenericSolver
from rl.base_solver import Solver as RLSolver
from core.problem_classes.problem_instance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int = 5000,
        train_step: int = 50000,
        evaluation_step: int = 10000,
        method: str = "A2C",
    ):
        """Initialize the A2C solver for one problem or compatible problems."""
        super().__init__(
            problem,
            max_step,
            train_step,
            evaluation_step,
            method,
        )

    def run(self):
        """Train the A2C model and return the solver."""
        return super().run()
