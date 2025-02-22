from utils.generic_solver import GenericSolver
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from solvers.ILP_arc import Solver as ILPSolver

import time
import networkx as nx
import numpy as np


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.service_package = self.problem_instance.service_package

        self.network_graph = self.problem_instance.network_graph
        self.nx_graph: nx.DiGraph = self.network_graph.nx_graph

        self.solver_name = "heuristic_ilp_arc"

        self._model_parameters()

    def _model_parameters(self):
        # Service parameters
        self.number_of_services = self.problem_instance.number_of_service
        self.demand_per_service = self.problem_instance.demand_per_service
        self.total_number_of_demands = self.problem_instance.total_number_of_demands
        self.needed_resources = self.problem_instance.needed_resources

        self.origins = self.problem_instance.origins
        self.min_flows = self.problem_instance.min_flows
        self.max_flows = self.problem_instance.max_flows
        self.latencies = self.problem_instance.latencies

        # Graph parameters
        self.arc_index_dictionary = self.network_graph.arc_index_dictionary
        self.arc_list = self.network_graph.arc_list
        self.alphas = self.network_graph.alphas
        self.betas = self.network_graph.betas
        self.node_number = self.nx_graph.number_of_nodes()
        self.edge_number = self.nx_graph.number_of_edges()
        self.capacities = self.network_graph.capacities
        self.available_resources = self.network_graph.available_resources

        self.cumulative_demands: np.ndarray = None

    def run(
        self,
    ):
        # Plan du solver:
        # - Attribuer les services a des serveurs
        # - Résoudre le problème de routage (avec arc/path methods)
        # ça revient à présolve une partie du problème et fixer ensuite des variables dans ILP arc et ILP path

        # Go presolve le probleme
        start_time = time.time()

        biggest_server = np.argmax(self.available_resources)
        service_server_association = [
            biggest_server for _ in range(self.nx_graph.number_of_nodes())
        ]

        ilp_solver_arc = ILPSolver(self.problem_instance)
        ilp_solver_arc._model_parameters()
        ilp_solver_arc._build_model()
        ilp_solver_arc._build_sets()
        ilp_solver_arc._build_variables()

        # Fixer ici la variable y
        for service_index in range(self.number_of_services):
            for server_index in range(self.nx_graph.number_of_nodes()):
                if server_index == service_server_association[service_index]:
                    ilp_solver_arc.model.y[service_index, server_index].fix(1)
                else:
                    ilp_solver_arc.model.y[service_index, server_index].fix(0)

        ilp_solver_arc.set_heuristic_solution()

        self.score = np.sum(ilp_solver_arc.q_var)
        self.runtime = time.time() - start_time
        self.build_time = ilp_solver_arc.build_time
