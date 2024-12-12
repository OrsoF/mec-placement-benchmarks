"Solveur RL PPO"

from utils.generic_solver import GenericSolver
from utils.generic_rl_solver import Solver as RLSolver
from problem_classes.probleminstance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem_instance,
        training_instances_database,
        max_step,
        train_step,
        evaluation_step,
        method='PPO',
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