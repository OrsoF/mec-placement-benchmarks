"Le solveur RL qu'on utilise."

from utils.generic_solver import GenericSolver
from problem_classes.probleminstance import ProblemInstance
import time
from utils.generic_solver import GenericSolver
from stable_baselines3 import A2C, PPO
from utils.gym_instance import SPPEnv
from tqdm import trange, tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


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
        self.problem_instance = problem_instance
        self.train_step = train_step
        self.solver_name = f"rl_{method}"
        # self.evaluation_step = evaluation_step

        # def make_env():
        #     env = SPPEnv(
        #         main_problem_instance=problem_instance,
        #         max_step=max_step,
        #         training_set=training_instances_database,
        #     )
        #     env.reset()
        #     return env

        # n_envs = 4
        # self.vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])

        self.env = SPPEnv(
            main_problem_instance=problem_instance,
            max_step=max_step,
            training_set=training_instances_database,
        )
        self.env.reset()

        if method == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=0, seed=0)
        elif method == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                seed=0,
                ent_coef=0.01,
            )
        else:
            assert False, "Not a known method."

    def run(
        self,
    ):
        start_time = time.time()

        self.model.learn(total_timesteps=self.train_step, progress_bar=True)

        self.runtime = time.time() - start_time
        self.score = max(self.env.scores)

        # obs, _ = self.env.reset(initial_instance=True)
        # action, _ = self.model.predict(obs)
        # for _ in trange(self.evaluation_step):
        #     obs, _, _, _, _ = self.env.step(action)
        #     action, _ = self.model.predict(obs)
