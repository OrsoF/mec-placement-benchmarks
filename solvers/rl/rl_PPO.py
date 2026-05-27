"Solveur RL PPO"

from core.generic_solver import GenericSolver
from rl.generic_rl_solver import Solver as RLSolver
from core.problem_classes.probleminstance import ProblemInstance


class Solver(RLSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        training_instances_database=None,
        # max_step=5000,
        # train_step=50000,
        # evaluation_step=5000,
        max_step=1000,
        train_step=10000,
        evaluation_step=100,
        method="PPO",
    ):
        max_step = 1000 + problem_instance.number_of_service * 10
        train_step = 10000 + problem_instance.number_of_service * 20
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
