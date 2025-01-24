"Le solveur RL qu'on utilise."

from utils.generic_solver import GenericSolver
from problem_classes.probleminstance import ProblemInstance
import time
from utils.generic_solver import GenericSolver
from stable_baselines3 import A2C, PPO
from utils.gym_instance import SPPEnv
from tqdm import trange, tqdm


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        training_instances_database: list,
        max_step: int,
        train_step: int,
        evaluation_step: int,
        method: str,
    ):
        self.solver_name = f"rl_{method}"
        self.evaluation_step = evaluation_step

        self.env = SPPEnv(
            main_problem_instance=problem_instance,
            max_step=max_step,
            training_set=training_instances_database,
        )
        self.env.reset()

        if method == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=0, seed=0)
        elif method == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=0, seed=0)
        else:
            assert False, "Not a known method."

        self.model.learn(total_timesteps=train_step, progress_bar=True)

    def run(
        self,
    ):
        start_time = time.time()

        obs, _ = self.env.reset(initial_instance=True)
        action, _ = self.model.predict(obs)
        for _ in trange(self.evaluation_step):
            obs, _, _, _, _ = self.env.step(action)
            action, _ = self.model.predict(obs)

        self.runtime = time.time() - start_time
        self.score = max(self.env.scores)
