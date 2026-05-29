"Solveur RL PPO"

from core.generic_solver import GenericSolver
from rl.base_solver import Solver as RLSolver
from core.problem_classes.problem_instance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        # max_step=5000,
        # train_step=50000,
        # evaluation_step=5000,
        max_step: int = 1000,
        train_step: int = 10000,
        evaluation_step: int = 100,
        method: str = "PPO",
    ):
        """Initialize the PPO solver for one problem or compatible problems."""
        problem_instance = problem[0] if isinstance(problem, list) else problem
        max_step = 1000 + problem_instance.number_of_service * 10
        train_step = 10000 + problem_instance.number_of_service * 20
        super().__init__(
            problem,
            max_step,
            train_step,
            evaluation_step,
            method,
        )

    def run(self):
        """Train the PPO model and return the solver."""
        return super().run()
