from utils.generic_solver import GenericSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import get_info_from_results
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc
from utils.data_management import get_result_informations
import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os
import numpy as np
from utils.generic_solver import GenericSolver
from stable_baselines3 import A2C, SAC, PPO
from utils.SPPGym import SPPGym


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.nx_graph = self.problem_instance.network_graph.nx_graph

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

        start_time = time.time()

        env = SPPGym(problem_instance=self.problem_instance)
        env.reset()
        model = PPO("MlpPolicy", env, verbose=0, seed=0)
        model.learn(total_timesteps=100000, progress_bar=False)

        self.runtime = time.time() - start_time
        self.score = env.max_q_reached
        self.x_variable, self.y_variable, self.q_variable = env.best_solution
