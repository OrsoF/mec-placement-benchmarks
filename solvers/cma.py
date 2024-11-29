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
from stable_baselines3 import A2C
from utils.SPPGym import SPPGym
import cma


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
        max_value = 0

        def objective(value: np.ndarray):
            global max_value
            S = self.problem_instance.number_of_service
            D = self.problem_instance.total_number_of_demands
            assert value.shape[0] == S + 2 * D
            env.state = value
            env.clip_state()
            try:
                env._test_solution()
                obj = np.sum(env.state[-D:])
                if obj > max_value:
                    max_value = obj

                return -obj
            except AssertionError:
                return 0.0

        env.reset()
        x0 = np.array(env.state).astype(np.float64)
        print(objective(x0))

        es = cma.CMAEvolutionStrategy(x0, 2)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [objective(x) for x in solutions])
            es.logger.add()  # write data to disc to be plotted
            es.disp()

        print("Meilleure solution CMA: {}".format(max_value))

        self.runtime = time.time() - start_time
        self.score = env.max_q_reached
        self.x_variable, self.y_variable, self.q_variable = env.best_solution
