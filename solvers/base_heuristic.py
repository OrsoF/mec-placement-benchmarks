import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from problem_classes.service import Service
from itertools import islice
from problem_classes.probleminstance import ProblemInstance
from utils.instance_solver import InstanceSolver


def greedy_server_allocation(
    resources: list[float], needed_resources: list[float], priority: list[float]
):
    assert len(priority) == len(
        resources
    ), "The priority factors is not shaped as list[server_number]."
    assert np.all([priority_value > 0 for priority_value in priority])

    # Initialize a list to store the allocations
    allocations = []

    # Create a list of server indices sorted by priority in descending order
    server_indices_sorted_by_priority = sorted(
        range(len(priority)),
        key=lambda server_index: resources[server_index] / priority[server_index],
        reverse=True,
    )

    # Iterate through each service
    for service_resources in needed_resources:
        best_server = None

        # Iterate through the sorted servers by priority
        for server_index in server_indices_sorted_by_priority:
            server_resources = resources[server_index]

            # Check if the server has enough resources for the service
            if server_resources >= service_resources:
                if best_server is None:
                    best_server = server_index

        if best_server is not None:
            # Allocate the service to the best server
            allocations.append(best_server)
            # Update available resources of the chosen server
            resources[best_server] -= service_resources
        else:
            # No suitable server found for the service
            allocations.append(None)

    assert None not in allocations, "One service has no server."
    assert len(allocations) == len(
        needed_resources
    ), "Allocations should be shaped as the number of service to place."

    return allocations


def server_allocations_to_matrix(allocations: list[int], number_of_nodes: int):
    service_node_matrix = np.zeros((len(allocations), number_of_nodes))
    for service_index, server in enumerate(allocations):
        service_node_matrix[service_index, server] = 1.0
    return service_node_matrix


class Solver(InstanceSolver):
    def __init__(self, problem_instance: ProblemInstance, limit_shortest_path: int = 5):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.nx_graph: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.limit_shortest_path = limit_shortest_path
        self.solver_name = "base_heuristic"

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
        self.resources = [service.needed_resources[0] for service in self.services]
        self.alpha = [
            self.nx_graph.edges[edge]["alpha"] for edge in self.nx_graph.edges
        ]
        self.beta = [self.nx_graph.edges[edge]["beta"] for edge in self.nx_graph.edges]

    def serv_number(self, d: int) -> tuple[int, int]:
        """
        Considering a list of services a the index of a demand d,
        the function returns the associated service index
        and the index of the demand in the considered service.
        """
        task_number = np.array(
            [
                sum(self.demand_per_service[:s])
                <= d
                < sum(self.demand_per_service[: s + 1])
                for s in range(len(self.services) - 1)
            ]
            + [1]
        )
        s = np.argmax(task_number)
        return s, d - sum(self.demand_per_service[:s])

    def capacity_check(self, path: list[int], quantity: float) -> bool:
        """
        Returns True if the quantity can pass through the path, False otherwise.
        """
        if len(path) == 1:
            return True
        else:
            return bool(
                min(
                    [
                        self.nx_graph.edges[(path[i], path[i + 1])][
                            "remaining_capacity"
                        ]
                        >= quantity
                        for i in range(len(path) - 1)
                    ]
                )
            )

    def latency_check(self, path: list[int], latency: float, quantity: float) -> bool:
        """
        Return True if the considered path can support
        an additional latency associated with quanity, False otherwise.
        """
        if len(path) == 1:
            return True
        else:
            used_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            current_latency = sum(
                self.nx_graph.edges[arc]["latency"] for arc in used_edges
            )
            additional_latency = sum(
                self.alpha[self.arc_dic[edge]] * quantity
                + self.beta[self.arc_dic[edge]]
                * int(self.nx_graph.edges[edge]["latency"] == 0)
                for edge in used_edges
            )
            return current_latency + additional_latency <= latency

    def services_nodes_association(
        self, weight_nodes: list[float]
    ) -> (np.ndarray, list[int]):
        """
        Given services, it associate a node to each service.
        """

        available_resources = [
            self.nx_graph.nodes[node]["resources"] for node in self.nx_graph.nodes
        ]
        needed_resources = np.copy(self.resources)
        assert len(needed_resources) == len(self.services)
        service_node_matrix = np.zeros((self.service_number, self.node_number))
        possible_nodes = [
            res / weight_nodes[i] for i, res in enumerate(available_resources)
        ]
        assert len(possible_nodes) == len(self.nx_graph.nodes)
        chosen_nodes = [-1] * self.service_number
        while max(needed_resources) > 0:
            s = np.argmax(needed_resources)
            resources = needed_resources[s]
            node = np.argmax(possible_nodes)
            if available_resources[node] - resources >= 0:
                available_resources[node] -= resources
                possible_nodes[node] = available_resources[node] / weight_nodes[node]
                needed_resources[s] = 0

                service_node_matrix[s, node] = 1
                chosen_nodes[s] = node
            else:
                weight_nodes[node] += 1
                possible_nodes = [
                    res / weight_nodes[i] for i, res in enumerate(available_resources)
                ]

        # available_resources = [
        #     self.nx_graph.nodes[node]["resources"] for node in self.nx_graph.nodes
        # ]
        # allocations = greedy_server_allocation(
        #     available_resources, np.copy(self.resources), weight_nodes
        # )
        # service_node_matrix = server_allocations_to_matrix(
        #     allocations, self.node_number
        # )

        return service_node_matrix, chosen_nodes

    def shortest_paths_choice(
        self,
        chosen_nodes: list[int],
        demand_arc_transit: np.ndarray,
        weight_edges: list[float],
        limit_shortest_paths,
    ) -> (np.ndarray, list, bool):
        """
        Make a choice between the available shortest paths to route the tasks.
        """
        for i, edge in enumerate(self.nx_graph.edges):
            self.nx_graph.edges[edge]["weight"] = weight_edges[i]

        service_number = np.array(
            [self.serv_number(d) for d in range(self.total_demands)]
        )

        used_paths = [[]] * self.total_demands
        for d in range(self.total_demands):
            s, d0 = service_number[d]
            origin, destination = self.origins[d], chosen_nodes[s]
            shortest_paths = islice(
                nx.algorithms.simple_paths.shortest_simple_paths(
                    self.nx_graph, origin, destination, weight="weight"
                ),
                limit_shortest_paths,
            )
            quantity, latency = self.min_flows[d], self.latencies[d]
            for i, path in enumerate(shortest_paths):
                if self.capacity_check(path, quantity) and self.latency_check(
                    path, latency, quantity
                ):
                    used_paths[d] = path
                    for i in range(len(path) - 1):
                        arc = (path[i], path[i + 1])
                        demand_arc_transit[d, self.arc_dic[arc]] = 1
                        self.nx_graph.edges[arc]["remaining_capacity"] -= quantity
                        self.nx_graph.edges[arc]["latency"] += (
                            self.alpha[self.arc_dic[arc]] * quantity
                            + self.beta[self.arc_dic[arc]]
                        )
                    flag = True
                    break
                else:
                    flag = False
            if not flag:
                return demand_arc_transit, used_paths, False
        return demand_arc_transit, used_paths, True

    def increase_quantities(
        self,
        used_paths: list[list[tuple]],
        weight_edges: list,
        flows: list,
    ):
        """
        Increase the quantity of each service progressively.
        """
        paths_len = [
            sum(
                weight_edges[self.arc_dic[(path[i], path[i + 1])]]
                for i in range(len(path) - 1)
            )
            for path in used_paths
        ]
        considered_demand = np.argmin(paths_len)
        quantity, path, latency = (
            self.min_flows[considered_demand],
            used_paths[considered_demand],
            self.latencies[considered_demand],
        )
        while True:
            if (
                self.capacity_check(path, quantity + 1)
                and self.latency_check(path, latency, quantity + 1)
                and quantity + 1 <= self.max_flows[considered_demand]
            ):
                flows[considered_demand] += 1
                quantity += 1
                path = used_paths[considered_demand]
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    self.nx_graph.edges[edge]["latency"] += self.alpha[
                        self.arc_dic[edge]
                    ]
                    self.nx_graph.edges[edge]["remaining_capacity"] -= 1
            else:
                paths_len[considered_demand] = np.inf
                considered_demand = np.argmin(paths_len)
                quantity, path, latency = (
                    self.min_flows[considered_demand],
                    used_paths[considered_demand],
                    self.latencies[considered_demand],
                )
            if np.min(paths_len) == np.inf:
                break
        return self.nx_graph, flows

    def run(self, options={"time": 3600, "threads": 1}):
        start_time = time.time()

        self._model_parameters()

        def heuristic_solve(
            weight_nodes: list,
            weight_edges: list,
            limit_shortest_paths: int,
            verbose: bool = False,
        ):
            if verbose:
                print(
                    "pmin",
                    sum(
                        demand["pmin"]
                        for service in self.services
                        for demand in service
                    ),
                )
                print(
                    "pmax",
                    sum(
                        demand["pmax"]
                        for service in self.services
                        for demand in service
                    ),
                )

            demand_arc_transit, y, p = (
                np.zeros((self.total_demands, self.edge_number)),
                np.zeros((self.service_number, self.node_number)),
                np.zeros(self.total_demands),
            )

            for arc in self.nx_graph.edges:
                self.nx_graph.edges[arc]["remaining_capacity"] = self.nx_graph.edges[
                    arc
                ]["capacity"]
                self.nx_graph.edges[arc]["latency"] = 0

            p = self.min_flows.copy()

            # Association noeuds/services

            service_node_matrix, chosen_nodes = self.services_nodes_association(
                weight_nodes
            )

            # For a given demand, we list the paths (shortest to longest) and take the first possible

            demand_arc_transit, used_paths, solved = self.shortest_paths_choice(
                chosen_nodes,
                demand_arc_transit,
                weight_edges,
                limit_shortest_paths,
            )
            if not solved:
                return demand_arc_transit, y, p, False

            # We now increase the quantity of data

            else:
                G, p = self.increase_quantities(
                    used_paths,
                    weight_edges,
                    p,
                )
                return demand_arc_transit, y, p, True

        weight_nodes, weight_edges = [1] * len(self.nx_graph.nodes), [1] * len(
            self.nx_graph.edges
        )

        a, b, c, d = heuristic_solve(
            weight_nodes,
            weight_edges,
            self.limit_shortest_path,
        )

        self.runtime = time.time() - start_time
        self.objective_value = np.sum(c)
