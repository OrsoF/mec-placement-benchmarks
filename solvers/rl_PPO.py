"Solveur RL PPO"

from utils.generic_solver import GenericSolver
from utils.generic_rl_solver import Solver as RLSolver
from problem_classes.probleminstance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem_instance,
        training_instances_database=None,
        # max_step=5000,
        # train_step=50000,
        # evaluation_step=5000,
        max_step=500,
        train_step=5000,
        evaluation_step=100,
        method="PPO",
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
