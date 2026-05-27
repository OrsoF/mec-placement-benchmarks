"Le solveur RL qu'on utilise."

from core.generic_solver import GenericSolver
from core.problem_classes.probleminstance import ProblemInstance
import time
from stable_baselines3 import A2C, PPO
from rl.gym_instance import SPPEnv


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        training_instances_database: list,
        max_step: int,
        train_step: int,
        evaluation_step: int,
        method: str,
        options: bool = False,
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
            options=options,
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
