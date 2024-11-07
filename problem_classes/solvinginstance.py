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
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.graph = self.problem_instance.network_graph.nx_graph.copy()

        self.service_placement: list = [
            -1 for _ in range(self.problem_instance.number_of_service)
        ]
        self.paths_used: list = [
            [] for _ in range(self.problem_instance.total_number_of_demands)
        ]
        self.quantities = [
            -1 for _ in range(self.problem_instance.total_number_of_demands)
        ]

        self._initialize_graph()

    def reset_instance(self):
        """Allows to reset the solving instance."""
        self.graph = self.problem_instance.network_graph.nx_graph.copy()
        self.service_placement: list = [
            -1 for _ in range(self.problem_instance.number_of_service)
        ]
        self.paths_used: list = [
            [] for _ in range(self.problem_instance.total_number_of_demands)
        ]
        self.quantities = [
            -1 for _ in range(self.problem_instance.total_number_of_demands)
        ]
        self._initialize_graph()

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
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["alpha"]

    def place_service(self, service_index: int, node: int, force: bool = False):
        assert self.service_placement[service_index] == -1, "Already placed"
        assert service_index < self.problem_instance.number_of_service
        assert node < self.graph.number_of_nodes()

        self.service_placement[service_index] = node
        self.graph.nodes[node][
            "remaining_resources"
        ] -= self.problem_instance.needed_resources[service_index]

        if not force:
            assert (
                self.graph.nodes[node]["remaining_resources"] >= 0
            ), "Not enough resources."

    def place_service_variable(self, y_variable: np.ndarray):
        for service_index in range(self.problem_instance.number_of_service):
            node_placement = np.argmax(y_variable[service_index, :])
            self.place_service(service_index, node_placement)

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

        if len(path):
            assert path[0] == origin and path[-1] == destination, "Path is not well set"

        self.paths_used[demand_index] = path
        self.quantities[demand_index] = quantity

        for edge in path_node_to_path_tuple(path):
            self.graph.edges[edge]["remaining_capacity"] -= quantity
            self.graph.edges[edge]["latency"] += (
                self.graph.edges[edge]["alpha"] * quantity
            )

            if not force:
                assert self.graph.edges[edge]["remaining_capacity"] >= 0

        if not force:
            latency_demand = sum(
                self.graph.edges[edge]["latency"]
                for edge in path_node_to_path_tuple(path)
            )
            assert latency_demand <= self.problem_instance.latencies[demand_index]

    def route_demand_variable(self, x_variable: np.ndarray, q_variable: np.ndarray):
        for demand_index in range(self.problem_instance.total_number_of_demands):
            edges_used_indices = np.nonzero(x_variable[demand_index])[0]
            edges_used = [
                self.problem_instance.network_graph.arc_list[index]
                for index in edges_used_indices
            ]
            start, end = (
                self.problem_instance.origins[demand_index],
                self.service_placement[
                    self.problem_instance.demand_to_service[demand_index]
                ],
            )
            path = edges_to_path(edges_used, start, end)
            self.route_demand(demand_index, q_variable[demand_index], path)

    def basic_placement_and_routing(self):
        cloud_node = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )
        for service_index in trange(self.problem_instance.number_of_service):
            self.place_service(service_index, cloud_node)

        for demand_index in trange(self.problem_instance.total_number_of_demands):
            origin = self.problem_instance.origins[demand_index]
            path = nx.shortest_path(self.graph, origin, cloud_node)
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
            self.graph.edges[edge]["latency"] -= (
                self.graph.edges[edge]["alpha"] * difference
            )

            if not force:
                assert (
                    self.graph.edges[edge]["remaining_capacity"] >= 0
                ), "Capacity exceeded."

        if not force:
            problem_latency = self.compute_demand_latency(demand_index)
            max_latency = self.problem_instance.latencies[demand_index]
            message = "Latency exceeded {} > {}".format(
                problem_latency,
                max_latency,
            )
            assert problem_latency <= max_latency, message

    def compute_demand_latency(self, demand_index: int) -> float:
        """
        For a given demand, return the associated graph latency.
        """
        path = path_node_to_path_tuple(self.paths_used[demand_index])
        return float(sum(self.graph.edges[edge]["latency"] for edge in path))

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
        """For a given service, replace it from its old node to new_node."""
        self.unplace_service(service_index)
        self.place_service(service_index, new_node, force)

        for demand_index in range(self.problem_instance.total_number_of_demands):
            if self.problem_instance.demand_to_service[demand_index] == service_index:
                self.route_demand(
                    demand_index, 1.0, new_demand_paths[demand_index], force
                )

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

    def set_variable_solution(
        self, x_var: np.ndarray, y_var: np.ndarray, q_var: np.ndarray
    ):
        self.place_service_variable(y_var)
        self.route_demand_variable(x_var, q_var)

    def get_score(self) -> float:
        return sum(self.quantities)

    def get_real_latency(self, demand_index):
        path = self.paths_used[demand_index]
        return sum(
            self.graph.edges[edge]["latency"] for edge in path_node_to_path_tuple(path)
        )

    def reset_routing(self):
        self.paths_used: list = [
            [] for _ in range(self.problem_instance.total_number_of_demands)
        ]
        self.quantities = [
            -1 for _ in range(self.problem_instance.total_number_of_demands)
        ]
        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["alpha"]

    def reset_placement(self):
        self.service_placement: list = [
            -1 for _ in range(self.problem_instance.number_of_service)
        ]
        for node in self.graph.nodes:
            self.graph.nodes[node]["remaining_resources"] = self.graph.nodes[node][
                "available_resources"
            ]

    def reset_instance(self):
        self.reset_routing()
        self.reset_placement()

    def get_demands_using_arc(self, arc: tuple) -> list:
        """Return the indices of the demands that uses the given arc."""
        return [
            demand_index
            for demand_index in range(self.problem_instance.total_number_of_demands)
            if arc in path_node_to_path_tuple(self.paths_used[demand_index])
        ]

    def get_crossing_demands(self, path: list) -> list:
        """Return the indices of the demands that use any arc of the given path."""
        if not len(path):
            return []
        if isinstance(path[0], int):
            path = path_node_to_path_tuple(path)
        demands = []
        for arc in path:
            demands += self.get_demands_using_arc(arc)
        demands = list(set(demands))
        return demands

    def get_remaining_quantity_possible(self, demand_index: int):
        """
        Les chemins de demandes peuvent s'intersecter !
        """
        path = path_node_to_path_tuple(self.paths_used[demand_index])
        remaining_flow = (
            self.problem_instance.max_flows[demand_index]
            - self.quantities[demand_index]
        )

        if not len(path):
            return remaining_flow
        else:
            distance_to_max_latency = float(
                self.problem_instance.latencies[demand_index]
                - self.get_real_latency(demand_index)
            )
            assert distance_to_max_latency >= 0.0, "Latency excedeed."
            max_alpha = max(self.graph.edges[edge]["alpha"] for edge in path)
            remaining_quantity_latency = np.floor(
                distance_to_max_latency / max_alpha / len(path)
            )
            assert remaining_quantity_latency >= 0.0, "Should not be seen."

            remaining_quantity_capacity = min(
                self.graph.edges[edge]["remaining_capacity"] for edge in path
            )
            assert remaining_quantity_capacity >= 0
            assert remaining_flow >= 0

            crossing_demands = self.get_crossing_demands(self.paths_used[demand_index])

            remaining_quantities_crossing_demands = min(
                [
                    (
                        self.problem_instance.latencies[demand_index]
                        - self.compute_demand_latency(demand_index)
                    )
                    / max_alpha
                    / len(self.paths_used[demand_index])
                    for demand_index in crossing_demands
                ]
            )

            remaining_quantity = int(
                min(
                    remaining_quantity_capacity,
                    remaining_quantity_latency,
                    remaining_flow,
                    remaining_quantities_crossing_demands,
                )
            )
            return remaining_quantity

    def _maximize_quantities_greedy(self):
        remaining_quantities = [
            self.get_remaining_quantity_possible(demand_index)
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]
        while max(remaining_quantities) >= 1:
            demand_index = np.argmax(remaining_quantities)
            new_quantity = (
                remaining_quantities[demand_index] + self.quantities[demand_index]
            )
            self.change_demand_quantity(demand_index, new_quantity)
            remaining_quantities = [
                self.get_remaining_quantity_possible(demand_index)
                for demand_index in range(self.problem_instance.total_number_of_demands)
            ]

    def get_actual_state(self) -> np.ndarray:
        state = self.service_placement + self.quantities
        return np.array(state)

    def get_state_dimension(self):
        return self.get_actual_state().shape[0]
