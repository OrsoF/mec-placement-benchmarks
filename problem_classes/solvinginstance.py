from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
import numpy as np
import networkx as nx
from utils.pickle import pickle_save
import matplotlib.pyplot as plt
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node


class SolvingInstance:
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.graph = self.problem_instance.network_graph.nx_graph

        self.service_placement: list = [
            -1 for _ in range(self.problem_instance.number_of_service)
        ]
        self.paths_used: list = [
            [] for _ in range(self.problem_instance.total_number_of_demands)
        ]
        self.quantities = [
            -1 for _ in range(self.problem_instance.total_number_of_demands)
        ]

        for node in self.graph.nodes:
            self.graph.nodes[node]["remaining_resources"] = self.graph.nodes[node][
                "resources"
            ]

        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["alpha"]

    def place_service(self, service_index: int, node: int, force: bool = False):
        assert self.service_placement[service_index] == -1, "Already placed"
        self.service_placement[service_index] = node
        self.graph[node][
            "remaining_resources"
        ] -= self.problem_instance.needed_resources[service_index]

        if not force:
            assert self.graph[node]["remaining_resources"] >= 0, "Not enough resources."

    def route_demand(
        self, demand_index: int, quantity: float, path: list, force: bool = False
    ):
        """Route given demands for the first time."""
        assert (
            not (len(self.paths_used[demand_index]))
            and self.quantities[demand_index] == -1
        ), "Demand already routed."
        service_index = self.problem_instance.demand_to_service[demand_index]

        assert self.service_placement[service_index] != -1, "Service not placed"

        origin = self.problem_instance.origins[demand_index]
        destination = self.service_placement[service_index]

        assert path[0] == origin and path[-1] == destination, "Path is not well set"

        self.paths_used[demand_index] = path
        self.quantities[demand_index] = quantity

        for edge in path_node_to_path_tuple(path):
            self.graph.edges[edge]["remaining_capacity"] -= quantity
            self.graph.edges[edge]["latency"] += (
                self.problem_instance.network_graph.alphas[edge] * quantity
            )

        if not force:
            assert self.graph.edges[edge]["remaining_capacity"] >= 0
            latency_demand = sum(
                self.graph.edges[edge]["latency"]
                for edge in path_node_to_path_tuple(path)
            )
            assert latency_demand <= self.problem_instance.latencies[demand_index]

    def basic_placement_and_routing(self):
        cloud_node = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )
        for service_index in range(self.problem_instance.number_of_service):
            self.place_service(service_index, cloud_node)

        for demand_index in range(self.problem_instance.total_number_of_demands):
            origin = self.problem_instance.origins[demand_index]
            path = nx.simple_paths.shortest_simple_paths(self.graph, origin, cloud_node)
            self.route_demand(demand_index, 1.0, path)

    def change_demand_quantity(
        self,
        demand_index: int,
        new_quantity: float,
        clip: bool = True,
        force: bool = False,
    ):
        if clip:
            new_quantity = max(
                self.problem_instance.min_flows[demand_index], new_quantity
            )
            new_quantity = min(
                self.problem_instance.max_flows[demand_index], new_quantity
            )

        difference = self.quantities[demand_index] - new_quantity

        self.quantities[demand_index] = new_quantity

        for edge in path_node_to_path_tuple(self.paths_used[demand_index]):
            self.graph.edges[edge]["remaining_capacity"] += difference
            self.graph.edges[edge]["latency"] -= self.graph.edges["alpha"] * difference

            if not force:
                assert self.graph.edges[edge]["remaining_capacity"] >= 0

        if not force:
            assert (
                self.compute_demand_latency(demand_index)
                <= self.problem_instance.latencies[demand_index]
            ), "Latency exceeded"

    def compute_demand_latency(self, demand_index: int):
        """
        For a given demand, return the associated graph latency.
        """
        path = path_node_to_path_tuple(self.paths_used[demand_index])
        return sum(self.graph.edges[edge]["latency"] for edge in path)

    def unroute_demand(self, demand_index: int):
        """
        Remove the associated latencies and capacities.
        """
        assert (
            self.quantities[demand_index] != -1 and self.paths_used[demand_index] != []
        ), "demand not routed"
        old_quantity = self.quantities[demand_index]
        self.quantities[demand_index] = -1

        old_path = self.paths_used[demand_index]
        self.paths_used[demand_index] = []

        for edge in path_node_to_path_tuple(old_path):
            self.graph.edges[edge]["remaining_capacity"] += old_quantity
            self.graph.edges[edge]["latency"] -= (
                self.graph.edges["alpha"] * old_quantity
            )

    def reroute_demand(self, demand_index: int, new_path: list, force: bool = False):
        """
        We assume that we reroute quantity.
        """
        self.unroute_demand(demand_index)
        self.route_demand(demand_index, 1.0, new_path, force)

    def unplace_service(self, service_index: int):
        for demand_index in range(self.problem_instance.total_number_of_demands):
            if self.problem_instance.demand_to_service[demand_index] == service_index:
                self.unroute_demand(demand_index)

        old_placement = self.service_placement[service_index]
        self.service_placement[service_index] = -1
        self.graph.nodes[old_placement][
            "remaining_resources"
        ] += self.problem_instance.needed_resources[service_index]

    def replace_service(
        self,
        service_index: int,
        new_node: int,
        new_demand_paths: list,
        force: bool = False,
    ):
        self.unplace_service(service_index)
        self.place_service(service_index, new_node, force)

        for demand_index in range(self.problem_instance.total_number_of_demands):
            if self.problem_instance.demand_to_service[demand_index] == service_index:
                self.route_demand(
                    demand_index, 1.0, new_demand_paths[demand_index], force
                )
