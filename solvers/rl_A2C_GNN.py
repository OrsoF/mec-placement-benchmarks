"Le solveur RL qu'on utilise."

from utils.generic_solver import GenericSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import get_info_from_results
from utils.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from utils.data_management import get_result_informations
import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os
import numpy as np
from utils.generic_solver import GenericSolver
from stable_baselines3 import A2C, PPO
from utils.SPPGym_GCN import SPPGymGNN
from tqdm import trange, tqdm
from utils.custom_policy import CustomActorCriticPolicy

class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        training_instances_database: list,
        max_step: int,
        train_step: int,
        evaluation_step: int,
    ):
        self.problem_instance = problem_instance
        self.nx_graph = self.problem_instance.network_graph.nx_graph
        self.training_instances_database = training_instances_database
        self.max_step = max_step
        self.train_step = train_step
        self.evaluation_step = evaluation_step

        self.solver_name = "rl_base"

    def run(
        self,
    ):
        """
        - Instantiate a gym environment
        Action space: (Service Placement, Shortest path indices, Quantity)
        - for each service, place it
        - Admit k-th SP
        - Choose quantities

        - Solve it with base RL
        """

        max_step = self.max_step
        train_step = self.train_step
        evaluation_step = self.evaluation_step

        env = SPPGymGNN(
            problem_instance=self.problem_instance,
            max_step=max_step,
            list_of_instances=self.training_instances_database,
        )
        env.reset()
        model = A2C(CustomActorCriticPolicy, env, verbose=0, seed=0)
        model.learn(total_timesteps=train_step, progress_bar=True)

        start_time = time.time()

        obs, _ = env.reset(initial_instance=True)
        action, _ = model.predict(obs)
        for _ in trange(evaluation_step):
            obs, _, _, _, _ = env.step(action)
            action, _ = model.predict(obs)

        self.runtime = time.time() - start_time
        self.score = max(env.scores[-evaluation_step:])
