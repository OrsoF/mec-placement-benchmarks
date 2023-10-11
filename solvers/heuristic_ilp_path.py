from utils.instance_solver import InstanceSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import get_info_from_results
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc
from solvers.base_heuristic import greedy_server_allocation
from solvers.ILP_path import Solver as ilp_solver

import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os
import numpy as np


class Solver(InstanceSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.nx_graph: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "heuristic_ilp_path"

        self._model_parameters()

    def _model_parameters(self):
        self.service_number = len(self.services)
        self.demand_per_service = [service.task_number for service in self.services]
        self.total_demands = sum(self.demand_per_service)
        self.node_number = self.nx_graph.number_of_nodes()
        self.edge_number = self.nx_graph.number_of_edges()
        self.arc_dic = {elem: i for i, elem in enumerate(self.nx_graph.edges)}
        self.arc_list = list(self.nx_graph.edges)
        self.origins = [
            origin for service in self.services for origin in service.origin_nodes
        ]
        self.min_flows = [
            min_flow for service in self.services for min_flow in service.min_flows
        ]
        self.max_flows = [
            max_flow for service in self.services for max_flow in service.max_flows
        ]
        self.latencies = [
            latency for service in self.services for latency in service.max_latencies
        ]
        self.needed_resources = [
            service.needed_resources[0] for service in self.services
        ]
        self.alpha = [
            self.nx_graph.edges[edge]["alpha"] for edge in self.nx_graph.edges
        ]
        self.beta = [self.nx_graph.edges[edge]["beta"] for edge in self.nx_graph.edges]
        self.available_resources = [
            self.problem_instance.network_graph.nx_graph.nodes[node]["resources"]
            for node in self.problem_instance.network_graph.nx_graph.nodes
        ]
        self.cumulative_demands: np.ndarray = None

    def run(self, options={"time": 60, "threads": 1}, solver: str = "gurobi"):
        start_time = time.time()

        server_priority = [1 for _ in range(self.nx_graph.number_of_nodes())]
        service_server_association = greedy_server_allocation(
            self.available_resources, self.needed_resources, server_priority
        )
        presolving_time = time.time() - start_time

        # Go solve le probleme ILP arc en fixant les variables y

        ilp_solver_arc = ilp_solver(self.problem_instance, maximum_path_number=3)
        ilp_solver_arc._model_parameters()
        ilp_solver_arc._build_model()
        ilp_solver_arc._build_sets()
        ilp_solver_arc._build_variables()

        # Fixer ici la variable y
        for service_index in range(self.service_number):
            for server_index in range(self.nx_graph.number_of_nodes()):
                if server_index == service_server_association[service_index]:
                    ilp_solver_arc.model.y[service_index, server_index].fix(1)
                else:
                    ilp_solver_arc.model.y[service_index, server_index].fix(0)

        ilp_solver_arc.run()

        self.runtime = float(ilp_solver_arc.runtime) + presolving_time
        self.lower_bound = ilp_solver_arc.lower_bound
        self.upper_bound = ilp_solver_arc.upper_bound
