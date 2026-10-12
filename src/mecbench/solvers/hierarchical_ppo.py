"""Hierarchical PPO for placement/routing followed by quantity optimization."""

import time

from stable_baselines3 import PPO

from mecbench.domain.problem import ProblemInstance
from mecbench.environments.hrl_env import FlowSetupEnv, PlacementRoutingEnv
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        problem: ProblemInstance,
        max_step: int | None = None,
        placement_routing_train_step: int | None = None,
        flow_train_step: int | None = None,
        max_path_candidates: int = 3,
        quantity_increment: int = 1,
        seed: int = 0,
    ):
        super().__init__(problem)
        self.solver_name = "hierarchical_ppo"
        self.seed = seed
        self.max_path_candidates = max_path_candidates
        self.quantity_increment = quantity_increment

        S = problem.number_of_service
        D = problem.total_number_of_demands
        self.max_step = max_step or 500 + 4 * (S + D)
        self.placement_routing_train_step = placement_routing_train_step or 5000 + 20 * (S + D)
        self.flow_train_step = flow_train_step or 3000 + 20 * D

        self.placement_routing_env = None
        self.flow_env = None
        self.placement_routing_model = None
        self.flow_model = None

    def _make_model(self, env, seed: int):
        return PPO(
            policy="MlpPolicy",
            env=env,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
        )

    def run(self):
        start = time.time()

        self.placement_routing_env = PlacementRoutingEnv(
            problem=self.problem_instance,
            max_step=self.max_step,
            max_path_candidates=self.max_path_candidates,
            warm_start_heuristic=True,
            quantity_increment=self.quantity_increment,
        )
        self.placement_routing_env.reset(seed=self.seed)
        self.placement_routing_model = self._make_model(
            self.placement_routing_env,
            self.seed,
        )
        self.placement_routing_model.learn(
            total_timesteps=self.placement_routing_train_step,
            progress_bar=False,
        )

        placement_routing_state = self.placement_routing_env.best_state_record[0]
        self.flow_env = FlowSetupEnv(
            problem=self.problem_instance,
            max_step=self.max_step,
            max_path_candidates=self.max_path_candidates,
            warm_start_heuristic=False,
            fixed_state=placement_routing_state,
            quantity_increment=self.quantity_increment,
        )
        self.flow_env.reset(seed=self.seed + 1)
        self.flow_model = self._make_model(self.flow_env, self.seed + 1)
        self.flow_model.learn(
            total_timesteps=self.flow_train_step,
            progress_bar=False,
        )

        best_state = self.flow_env.best_state_record[0]
        self.x_var, self.y_var, self.q_var = self.flow_env._state_to_variables(best_state)
        self._set_output_variables(self.x_var, self.y_var, self.q_var)
        self.runtime = time.time() - start
        self.score = float(sum(self.q_var))
        self.build_time = 0.0
        return self
