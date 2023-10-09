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
    resources_in_servers: list[float],
    resources_needed_by_services: list[float],
    server_priority: list[float],
):
    """
    Associated a list of services with needed_resources to a list of server with resources following a given priority.
    """
    assert len(server_priority) == len(
        resources_in_servers
    ), "The priority factors is not shaped as list[server_number]."
    assert np.all([priority_value > 0 for priority_value in server_priority])

    # Initialize a list to store the allocations
    service_server_association = []

    # Create a list of server indices sorted by priority in descending order
    server_indices_sorted_by_priority = sorted(
        range(len(server_priority)),
        key=lambda server_index: resources_in_servers[server_index]
        / server_priority[server_index],
        reverse=True,
    )

    # Iterate through each service
    for service_resources in resources_needed_by_services:
        best_server = None

        # Iterate through the sorted servers by priority
        for server_index in server_indices_sorted_by_priority:
            server_resources = resources_in_servers[server_index]

            # Check if the server has enough resources for the service
            if server_resources >= service_resources:
                if best_server is None:
                    best_server = server_index

        if best_server is not None:
            # Allocate the service to the best server
            service_server_association.append(best_server)
            # Update available resources of the chosen server
            resources_in_servers[best_server] -= service_resources
        else:
            # No suitable server found for the service
            service_server_association.append(None)

    assert None not in service_server_association, "One service has no server."
    assert len(service_server_association) == len(
        resources_needed_by_services
    ), "Allocations should be shaped as the number of service to place."

    return service_server_association


def server_allocations_to_matrix(
    service_server_association: list[int], number_of_servers: int
):
    """
    Transform a list of size |services| with the number of the given server into a matrix of size (services, servers) with zeros and ones.
    """
    service_node_matrix = np.zeros((len(service_server_association), number_of_servers))
    for service_index, server in enumerate(service_server_association):
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

        self.cumulative_demands: np.ndarray = None

    def serv_number(self, d: int) -> tuple[int, int]:
        """
        Considering a list of services a the index of a demand d,
        the function returns the associated service index
        and the index of the demand in the considered service.
        """

        if self.cumulative_demands is None:
            self.cumulative_demands = np.cumsum(self.demand_per_service)
            self.cumulative_demands_plus_zero = np.array(
                [0] + list(self.cumulative_demands)
            )

        service = np.where(self.cumulative_demands_plus_zero <= d)[0][-1]

        local_demand_number = d - self.cumulative_demands_plus_zero[service]

        assert service >= 0, "Service number should be a non negative integer."
        assert (
            local_demand_number >= 0
        ), "Demand number should be a non negative integer."

        return service, local_demand_number

    def capacity_check(self, path: list[int], quantity: float) -> bool:
        """
        Returns True if the quantity can pass through the path, False otherwise.
        """
        if len(path) == 1:
            # The path is of length 0
            return True
        elif len(path) == 2:
            # The path is an edge, we check the condition on it.
            return self.nx_graph.edges[tuple(path)]["remaining_capacity"] >= quantity
        else:
            # The path is made of multiple edges, we check the condition on the first edge and on the the remaining path.
            return self.capacity_check(path[:2], quantity) and self.capacity_check(
                path[1:], quantity
            )

    def latency_check(self, path: list[int], latency: float, quantity: float) -> bool:
        """
        Return True if the considered service can support
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

    def services_nodes_association(self) -> (np.ndarray, list[int]):
        """
        Given services, it associate a node to each service.
        """

        available_resources = [
            self.nx_graph.nodes[node]["resources"] for node in self.nx_graph.nodes
        ]
        allocations = greedy_server_allocation(
            available_resources, np.copy(self.resources), self.weight_nodes
        )
        service_node_matrix = server_allocations_to_matrix(
            allocations, self.node_number
        )

        return service_node_matrix, allocations

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
            s, _ = service_number[d]
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

    def run(self, limit_shortest_paths: int = 2, options={"time": 3600, "threads": 1}):
        start_time = time.time()

        self.weight_nodes = [1] * len(self.nx_graph.nodes)
        self.weight_edges = [1] * len(self.nx_graph.edges)

        self._model_parameters()

        demand_arc_transit = np.zeros((self.total_demands, self.edge_number))
        quantity_matrix = np.zeros(self.total_demands)

        for arc in self.nx_graph.edges:
            self.nx_graph.edges[arc]["remaining_capacity"] = self.nx_graph.edges[arc][
                "capacity"
            ]
            self.nx_graph.edges[arc]["latency"] = 0

        p = self.min_flows.copy()

        # Association noeuds/services
        service_node_matrix, chosen_nodes = self.services_nodes_association()

        # For a given demand, we list the paths (shortest to longest) and take the first possible
        demand_arc_transit, used_paths, solved = self.shortest_paths_choice(
            chosen_nodes,
            demand_arc_transit,
            self.weight_edges,
            limit_shortest_paths,
        )

        if solved:
            _, p = self.increase_quantities(used_paths, self.weight_edges, p)

        self.runtime = time.time() - start_time
        self.objective_value = np.sum(p)
