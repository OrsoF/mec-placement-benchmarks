from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage
import numpy as np
import networkx as nx
from utils.pickle import pickle_save
import matplotlib.pyplot as plt
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node


class ProblemInstance:
    def __init__(
        self,
        network_graph: NetworkGraph,
        service_package: ServicePackage,
        problem_name: str,
    ) -> None:
        self.network_graph = network_graph
        self.service_package = service_package
        self.problem_name = problem_name

        self.number_of_service: int = self.service_package.number_of_services
        self.demand_per_service: list[int] = [
            service.task_number for service in self.service_package.services
        ]
        self.total_number_of_demands: int = sum(self.demand_per_service)

        self.origins = [
            origin
            for service in self.service_package.services
            for origin in service.origin_nodes
        ]
        self.min_flows = [
            min_flow
            for service in self.service_package.services
            for min_flow in service.min_flows
        ]
        self.max_flows = [
            max_flow
            for service in self.service_package.services
            for max_flow in service.max_flows
        ]
        self.latencies = [
            latency
            for service in self.service_package.services
            for latency in service.max_latencies
        ]
        self.needed_resources = [
            service.needed_resources for service in self.service_package.services
        ]
        self.demand_to_service = [
            service_index
            for service_index in range(self.number_of_service)
            for _ in range(self.demand_per_service[service_index])
        ]

        self.validate_instance()

    def _set_arc_solution(
        self, x_variable: np.ndarray, y_variable: np.ndarray, q_variable: np.ndarray
    ):
        """
        Checks the provided variables shape
        and sets them to the current solution.
        """
        assert x_variable.shape == (
            self.total_number_of_demands,
            self.network_graph.nx_graph.number_of_edges(),
        )
        assert y_variable.shape == (
            self.number_of_service,
            self.network_graph.nx_graph.number_of_nodes(),
        )
        assert q_variable.shape[0] == self.total_number_of_demands
        self.x_variable, self.y_variable, self.q_variable = (
            x_variable,
            y_variable,
            q_variable,
        )

    def print_current_solution(self, show_graph: bool = False):
        for service_index in range(self.number_of_service):  # Print service placement
            node_placement = np.nonzero(self.y_variable[service_index])[0][0]
            PLACEMENT_SERVICE = "Service {} is placed on node {}".format(
                service_index, node_placement
            )
            print(PLACEMENT_SERVICE)
        print()

        for demand_index in range(self.total_number_of_demands):  # Paths
            service_index = (
                demand_index // self.service_package.number_of_demand_per_service
            )
            demand_origin = self.origins[demand_index]
            demand_destination = np.nonzero(self.y_variable[service_index])[0][0]

            edge_used_indices = np.nonzero(self.x_variable[demand_index])[0]
            edge_used = [
                self.network_graph.arc_list[edge_index]
                for edge_index in edge_used_indices
            ]
            # print(
            #     "Origine : {}, Destination : {}".format(
            #         demand_origin, demand_destination
            #     )
            # )
            if not edge_used:
                print("Chemin utilisé : {}".format([demand_origin, demand_destination]))
            else:
                ordered_path = nx.dag_longest_path(nx.DiGraph(edge_used))
                print("Chemin utilisé : {}".format(ordered_path))
        print()

        # Latencies
        capacities_used = [
            0.0 for _ in range(self.network_graph.nx_graph.number_of_edges())
        ]
        for demand_index in range(self.total_number_of_demands):  # Paths
            service_index = (
                demand_index // self.service_package.number_of_demand_per_service
            )
            quantity = self.q_variable[demand_index]

            edge_used_indices = np.nonzero(self.x_variable[demand_index])[0]
            for edge_index in edge_used_indices:
                capacities_used[edge_index] += quantity

            latency = sum(
                self.network_graph.alphas[edge_index] * quantity
                + self.network_graph.betas[edge_index]
                for edge_index in edge_used_indices
            )

            LATENCY_PRINT = "Real latency {} <= Maximum {}".format(
                latency, self.latencies[demand_index]
            )
            print(LATENCY_PRINT)
        print()

        # Capacities
        for edge_index, capa in enumerate(capacities_used):
            CAPACITY_PRINT = "Capacity used {} <= Maximum {}".format(
                capa, self.network_graph.capacities[edge_index]
            )
            print(CAPACITY_PRINT)
        print()

        for demand_index in range(self.total_number_of_demands):
            print(
                "Quantity {} <= Maximum {}".format(
                    self.q_variable[demand_index], self.max_flows[demand_index]
                )
            )
        print()

        if show_graph:
            nx.draw(
                self.network_graph.nx_graph,
                with_labels=True,
                node_color="lightblue",
                node_size=2000,
                font_size=15,
                font_weight="bold",
                arrowsize=20,
            )
            plt.show()

    def check_current_solution(self):
        """
        In this function, we check the differents contraints:
        - Service assignment and resources on nodes
        - Edge capacities
        - Routing
        - Latencies
        Using the attributes self.x_variable, self.y_variable and self.q_variable.
        """
        assert (
            (self.y_variable.sum(axis=1) - 1) ** 2
        ).max() < 1e-2  # Service assignment

        for node in range(
            self.network_graph.nx_graph.number_of_nodes()
        ):  # Resources on each node
            used_resources = sum(
                [
                    self.y_variable[service_index, node]
                    * self.needed_resources[service_index]
                    for service_index in range(self.number_of_service)
                ]
            )
            assert used_resources <= self.network_graph.available_resources[node]

        assert np.all(
            self.x_variable.T.dot(self.q_variable)
            <= np.array(self.network_graph.capacities)
        ), f"Edge capacity exceeded : {self.x_variable.T.dot(self.q_variable)} <= {np.array(self.network_graph.capacities)}"  # Check edges capacities

        for demand_index in range(self.total_number_of_demands):  # Check routing
            if not len(
                np.nonzero(self.x_variable[demand_index, :])[0]
            ):  # Le chemin est de longueur 0, l'origine et la destination sont les mêmes
                continue
            else:
                service_index = self.demand_to_service[demand_index]
                demand_origin = self.origins[demand_index]
                demand_destination = np.argmax(self.y_variable[service_index])
                unordered_path = [
                    list(self.network_graph.nx_graph.edges)[edge_index]
                    for edge_index in np.nonzero(self.x_variable[demand_index, :])[0]
                ]
                ordered_path = nx.dag_longest_path(nx.DiGraph(unordered_path))
                assert (ordered_path[0] == demand_origin) and (
                    ordered_path[-1] == demand_destination
                )

        # Latencies
        capacities_used = [
            0.0 for _ in range(self.network_graph.nx_graph.number_of_edges())
        ]
        for demand_index in range(self.total_number_of_demands):  # Paths
            quantity = self.q_variable[demand_index]
            edge_used_indices = np.nonzero(self.x_variable[demand_index])[0]
            for edge_index in edge_used_indices:
                capacities_used[edge_index] += quantity
            latency = sum(
                self.network_graph.alphas[edge_index] * quantity
                + self.network_graph.betas[edge_index]
                for edge_index in edge_used_indices
            )
            assert latency <= self.latencies[demand_index]

        # Capacities
        for edge_index, capa in enumerate(capacities_used):
            assert capa <= self.network_graph.capacities[edge_index]

    def validate_instance(self):
        """
        Check if the basic placement and routing can be done.
        """
        placement_node = np.argmax(self.network_graph.available_resources)

        self.x_variable = np.zeros(
            (
                self.total_number_of_demands,
                self.network_graph.nx_graph.number_of_edges(),
            )
        )
        for demand_index in range(self.total_number_of_demands):
            demand_origin = self.origins[demand_index]
            path = nx.shortest_path(
                self.network_graph.nx_graph, demand_origin, placement_node
            )
            path = path_node_to_path_tuple(path)
            for edge in path:
                edge_index = self.network_graph.arc_index_dictionary[edge]
                self.x_variable[demand_index, edge_index] = 1.0

        placement = [
            1 if node_index == placement_node else 0
            for node_index in range(self.network_graph.nx_graph.number_of_nodes())
        ]
        self.y_variable = np.array([placement for _ in range(self.number_of_service)])

        self.q_variable = np.array(self.min_flows)
        self.check_current_solution()


class ToyInstance(ProblemInstance):
    def __init__(self) -> None:
        """Toy Instance"""
        pass
