"""Shared training wrapper for reinforcement-learning solvers."""

import time

from stable_baselines3 import A2C, PPO

from core.generic_solver import GenericSolver
from core.problem_classes.problem_instance import ProblemInstance
from rl.spp_env import SPPEnv


MODEL_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
}


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        training_instances_database: list[ProblemInstance] | None,
        max_step: int,
        train_step: int,
        evaluation_step: int,
        method: str,
        reuse_best_on_reset: bool = False,
    ):
        super().__init__(problem_instance)

        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.method = method.upper()
        self.solver_name = f"rl_{self.method}"

        self.env = self._make_env(
            problem_instance,
            max_step,
            training_instances_database,
            reuse_best_on_reset,
        )
        self.model = self._make_model()

    def _make_env(
        self,
        problem_instance: ProblemInstance,
        max_step: int,
        training_instances_database: list[ProblemInstance] | None,
        reuse_best_on_reset: bool,
    ) -> SPPEnv:
        env = SPPEnv(
            main_problem_instance=problem_instance,
            max_step=max_step,
            training_set=training_instances_database,
            reuse_best_on_reset=reuse_best_on_reset,
        )
        env.reset()
        return env

    def _make_model(self):
        if self.method not in MODEL_CLASSES:
            known_methods = ", ".join(sorted(MODEL_CLASSES))
            raise ValueError(
                f"Unknown RL method '{self.method}'. Expected one of: {known_methods}."
            )

        model_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
            "verbose": 0,
            "seed": 0,
        }
        if self.method == "PPO":
            model_kwargs["ent_coef"] = 0.01

        return MODEL_CLASSES[self.method](**model_kwargs)

    def run(self):
        start_time = time.time()

        self.model.learn(total_timesteps=self.train_step, progress_bar=False)

        self.runtime = time.time() - start_time
        self.score = max(self.env.served_quantity_history)
        self.build_time = 0.0
        return self
