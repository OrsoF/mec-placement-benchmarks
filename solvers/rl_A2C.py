"Solveur RL PPO"

from utils.generic_solver import GenericSolver
from utils.generic_rl_solver import Solver as RLSolver
from problem_classes.probleminstance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem_instance,
        training_instances_database=None,
        max_step=50,
        train_step=500,
        evaluation_step=100,
        method="A2C",
    ):
        super().__init__(
            problem_instance,
            training_instances_database,
            max_step,
            train_step,
            evaluation_step,
            method,
        )

    def run(self):
        return super().run()
