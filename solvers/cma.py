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
from utils.gym_instance import SPPEnv
import cma
from problem_classes.solvinginstance import SolvingInstance


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, max_pcc: int = 10):
        self.problem_instance = problem_instance
        self.nx_graph = self.problem_instance.network_graph.nx_graph
        self.max_pcc = max_pcc

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

        self.max_value = 0
        start_time = time.time()

        self.sol_ins = SolvingInstance(self.problem_instance)
        x0 = self.initial_value()

        es = cma.CMAEvolutionStrategy(x0, 2)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.objective(x) for x in solutions])
            es.logger.add()  # write data to disc to be plotted
            es.disp()

        # print("Meilleure solution CMA: {}".format(self.max_value))

        self.runtime = time.time() - start_time
        self.score = self.max_value

    def objective(self, value: np.ndarray):
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands

        placement = [
            max(0, min(int(elem), self.nx_graph.number_of_nodes() - 1))
            for elem in value[:S]
        ]
        routing = [max(0, min(int(elem), self.max_pcc)) for elem in value[S : S + D]]
        quantities = [
            max(1, min(int(elem), self.problem_instance.max_flows[0]))
            for elem in value[-D:]
        ]
        it_works = self.sol_ins.test_solution(placement, routing, quantities)

        if it_works:
            if sum(quantities) > self.max_value:
                self.max_value = sum(quantities)
            return -sum(quantities)
        else:
            return 0.0

    def initial_value(self) -> np.ndarray:
        placement, routing, quantities = self.sol_ins.basic_placement_and_routing()
        return np.array(placement + routing + quantities)
