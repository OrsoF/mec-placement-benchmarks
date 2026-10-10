"""Shared training wrapper for reinforcement-learning mecbench.solvers."""

import time

from stable_baselines3 import A2C, PPO

from mecbench.solvers.base import GenericSolver
from mecbench.domain.problem import ProblemInstance
from mecbench.environments.spp_env import SPPEnv


MODEL_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
}


class Solver(GenericSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int,
        train_step: int,
        evaluation_step: int,
        method: str,
        reuse_best_on_reset: bool = False,
        seed: int = 0,
    ):
        """Initialize the RL solver around one problem or compatible problems."""
        if isinstance(problem, list) and not problem:
            raise ValueError("RL Solver requires at least one problem instance.")
        problem_instance = problem[0] if isinstance(problem, list) else problem
        super().__init__(problem_instance)

        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.method = method.upper()
        self.seed = seed
        self.solver_name = f"rl_{self.method}"

        self.env = self._make_env(
            problem,
            max_step,
            reuse_best_on_reset,
        )
        self.model = self._make_model()

    def _make_env(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int,
        reuse_best_on_reset: bool,
    ) -> SPPEnv:
        """Create the Gym environment used by the SB3 model."""
        env = SPPEnv(
            problem=problem,
            max_step=max_step,
            reuse_best_on_reset=reuse_best_on_reset,
        )
        env.reset(seed=self.seed)
        return env

    def _make_model(self):
        """Create the configured Stable-Baselines3 model."""
        if self.method not in MODEL_CLASSES:
            known_methods = ", ".join(sorted(MODEL_CLASSES))
            raise ValueError(
                f"Unknown RL method '{self.method}'. Expected one of: {known_methods}."
            )

        model_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
            "verbose": 0,
            "seed": self.seed,
        }
        if self.method == "PPO":
            model_kwargs["ent_coef"] = 0.01

        return MODEL_CLASSES[self.method](**model_kwargs)

    def run(self):
        """Train the model and record aggregate solver metrics."""
        start_time = time.time()

        self.model.learn(total_timesteps=self.train_step, progress_bar=False)

        best_state = self.env.best_state_record[0]
        self.x_var, self.y_var, self.q_var = self.env._state_to_variables(best_state)
        self._set_output_variables(self.x_var, self.y_var, self.q_var)
        self.runtime = time.time() - start_time
        self.score = sum(self.q_var)
        self.build_time = 0.0
        return self


class A2CSolver(Solver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int = 5000,
        train_step: int = 50000,
        evaluation_step: int = 10000,
        method: str = "A2C",
        seed: int = 0,
    ):
        """Initialize the default A2C RL solver."""
        super().__init__(
            problem,
            max_step,
            train_step,
            evaluation_step,
            method,
            seed=seed,
        )


class PPOSolver(Solver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int | None = None,
        train_step: int | None = None,
        evaluation_step: int = 100,
        method: str = "PPO",
        reuse_best_on_reset: bool = False,
        seed: int = 0,
    ):
        """Initialize the default PPO RL solver."""
        problem_instance = problem[0] if isinstance(problem, list) else problem
        max_step = 1000 + problem_instance.number_of_service * 10 if max_step is None else max_step
        train_step = 10000 + problem_instance.number_of_service * 20 if train_step is None else train_step
        super().__init__(
            problem=problem,
            max_step=max_step,
            train_step=train_step,
            evaluation_step=evaluation_step,
            method=method,
            reuse_best_on_reset=reuse_best_on_reset,
            seed=seed,
        )


class PPOOptionsSolver(PPOSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int | None = None,
        train_step: int | None = None,
        evaluation_step: int = 100,
        method: str = "PPO",
        reuse_best_on_reset: bool = True,
        seed: int = 0,
    ):
        """Initialize the PPO solver variant that reuses the best reset state."""
        super().__init__(
            problem=problem,
            max_step=max_step,
            train_step=train_step,
            evaluation_step=evaluation_step,
            method=method,
            reuse_best_on_reset=reuse_best_on_reset,
            seed=seed,
        )
