from utils.generic_solver import GenericSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import get_result_informations
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc
from itertools import islice

from time import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "MDP_formulation"

    def run(
        self,
    ):
        # Formuler un MDP associté, à savoir : SATR
        self.state_list = []

        self.state_space, self.state_dim = 0, 0

        self.action_list = []
        self.action_space, self.action_dim = 0, 0

        # Résoudre ce MDP

        self.runtime, self.lower_bound, self.upper_bound = 0, 0, 0
