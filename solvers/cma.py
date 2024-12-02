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

        self.max_pcc = 3

        self.env = SPPGym(
            problem_instance=self.problem_instance,
            max_step=1000,
            max_pcc=self.max_pcc,
        )
        self.env.reset()
        self.max_value = 0

        self.env.reset()
        x0 = np.array(self.env.state).astype(np.float64)
        print(self.objective(x0))

        es = cma.CMAEvolutionStrategy(x0, 2)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.objective(x) for x in solutions])
            es.logger.add()  # write data to disc to be plotted
            es.disp()

        print("Meilleure solution CMA: {}".format(self.max_value))

        self.runtime = time.time() - start_time
        self.score = self.max_value

    def objective(self, value: np.ndarray):
        value = np.array(value)
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands
        assert value.shape[0] == S + 2 * D

        value[:S] = np.clip(value[:S], 0, self.nx_graph.number_of_nodes() - 1).astype(
            int
        )
        value[S : S + D] = np.clip(value[S : S + D], 0, self.max_pcc).astype(int)
        value[-D:] = np.clip(value[-D:], 1, self.problem_instance.max_flows[0]).astype(
            int
        )
        value = [int(elem) for elem in value]

        self.env.state = value
        
        try:
            assert self.env._test_solution(self.env.state)
            obj = np.sum(self.env.state[-D:])
            if obj > self.max_value:
                self.max_value = obj

            return -obj
        except AssertionError:
            return 0.0
