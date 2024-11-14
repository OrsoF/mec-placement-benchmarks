import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange

from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
from utils.pickle import pickle_save
from utils.graph_utils import (
    path_node_to_path_tuple,
    path_tuple_to_path_node,
    edges_to_path,
)


class SolvingInstance:
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True):
        self.problem_instance = problem_instance
        self.verbose = verbose
        self.graph = self.problem_instance.network_graph.nx_graph.copy()

        self._initialize_graph()

    def test_solution(
        self, service_placement: list, paths: list, quantities: list
    ) -> bool:
        # Idea:
        # Route demands, checks capacity constraints, resources, latencies

        # Place services
        for service_index in self.problem_instance.number_of_service:
            node = service_placement[service_index]
            self.graph.nodes[node][
                "remaining_resources"
            ] -= self.problem_instance.needed_resources[service_index]

        # Check resources constraints
        for node in self.graph.nodes:
            if self.graph.nodes[node]["remaining_resources"] < 0.0:
                if self.verbose:
                    print("Wrong resources allocation.")
                return False

        # Route demands
        for demand_index in range(self.problem_instance.total_number_of_demands):
            path = paths[demand_index]
            quantity = quantities[demand_index]
            for edge in path_node_to_path_tuple(path):
                self.graph.edges[edge]["remaining_capacity"] -= quantity
                self.graph.edges[edge]["latency"] += (
                    self.graph.edges[edge]["alpha"] * quantity
                )

        # Check latency constraint
        for demand_index in range(self.problem_instance.total_number_of_demands):
            path = paths[demand_index]
            demand_latency = sum(
            self.graph.edges[edge]["latency"] for edge in path_node_to_path_tuple(path)
            )
            if demand_latency > self.problem_instance.latencies[demand_index]:
                if self.verbose:
                    print("Latency not respected.")
                return False

        # Check capacity constraint
        for edge in self.graph.edges:
            if self.graph.edges[edge]["remaining_capacity"] < 0.0:
                if self.verbose:
                    print("Capacity condition not checked.")
                return False
        return True

    def _initialize_graph(self):
        """
        Initialize graph with trivial placement on cloud servers.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["remaining_resources"] = self.graph.nodes[node][
                "available_resources"
            ]

        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]

    def test_solution_variable(self, x_var, y_var, q_var) -> bool:
        service_placement = y_var

        paths = []
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            used_edges = [
                self.problem_instance.network_graph.arc_list[arc_index]
                for arc_index in np.nonzero(x_var)[0]
            ]
            start = self.problem_instance.origins[demand_index]
            end = y_var[service_index]
            path = edges_to_path(used_edges, start, end)
            paths.append(path)

        quantities = q_var

        return self.test_solution(service_placement, paths, quantities)

    def render_solution(self):
        for service_index in range(self.problem_instance.number_of_service):
            print(
                f"Service {service_index} placed on node {self.service_placement[service_index]}."
            )
        print()

        for node_index in self.graph.nodes:
            node = self.graph.nodes[node_index]
            print(
                "Node {} remaining resources : {} / {}".format(
                    node_index, node["remaining_resources"], node["available_resources"]
                )
            )
        print()

        for edge_tuple in self.graph.edges:
            edge = self.graph.edges[edge_tuple]
            print(
                "Edge {} remaining capacity : {} / {}".format(
                    edge_tuple, edge["remaining_capacity"], edge["capacity"]
                )
            )
        print()

        for demand_index in range(self.problem_instance.total_number_of_demands):
            print(
                "Path for demand {} : {}".format(
                    demand_index, self.paths_used[demand_index]
                )
            )
        print()

        for demand_index in range(self.problem_instance.total_number_of_demands):
            print(
                "Quantity for demand {} : {} / {}".format(
                    demand_index,
                    self.quantities[demand_index],
                    self.problem_instance.max_flows[demand_index],
                )
            )
        print()

        for demand_index in range(self.problem_instance.total_number_of_demands):
            latency = self.get_real_latency(demand_index)
            print(
                "Latency for demand {} : {} / {}".format(
                    demand_index, latency, self.problem_instance.latencies[demand_index]
                )
            )
        print()
        print("Solution reached : {}".format(sum(self.quantities)))

    def get_arc_variables(self):
        y_variable = np.zeros(
            (self.problem_instance.number_of_service, self.graph.number_of_nodes()),
            dtype=int,
        )
        for i, col_index in enumerate(self.service_placement):
            y_variable[i, col_index] = 1

        x_variable = np.zeros(
            (
                self.problem_instance.total_number_of_demands,
                self.graph.number_of_edges(),
            ),
            dtype=int,
        )
        for demand_index, path in enumerate(self.paths_used):
            for edge in path_node_to_path_tuple(path):
                edge_index = self.problem_instance.network_graph.arc_index_dictionary[
                    edge
                ]
                x_variable[demand_index, edge_index] = 1

        q_variable = np.array(self.quantities)

        return x_variable, y_variable, q_variable
