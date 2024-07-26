from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage
import numpy as np
import networkx as nx

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

    def set_arc_solution(self, x_variable:np.ndarray, y_variable:np.ndarray, q_variable:np.ndarray):
        assert x_variable.shape == (self.total_number_of_demands, self.network_graph.nx_graph.number_of_edges())
        assert y_variable.shape == (self.number_of_service, self.network_graph.nx_graph.number_of_nodes())
        assert q_variable.shape[0] == self.total_number_of_demands
        self.x_variable, self.y_variable, self.q_variable = x_variable, y_variable, q_variable

    def check_current_solution(self):
        """
        In this function, we check the differents contraints:
        - Service assignment and resources on nodes
        - Edge capacities
        - Routing
        - Latencies
        """
        assert np.linalg.norm(self.y_variable.sum(axis=1) - 1) < 1e-2 # Service assignment
        for node in range(self.network_graph.nx_graph.number_of_nodes()): # Resources on each node
            used_resources = [self.y_variable[service_index, node] * self.needed_resources[service_index] for service_index in range(self.number_of_service)]
            assert used_resources <= self.network_graph.available_resources[node]

        assert np.all(self.x_variable.T.dot(self.q_variable) <= np.array(self.network_graph.capacities)) # Check edges capacities

        for demand_index in range(self.total_number_of_demands): # Check routing
            service_index = demand_index % self.service_package.number_of_demand_per_service
            demand_origin = self.origins[demand_index]
            demand_destination = np.nonzero(self.y_variable[service_index])
            unordered_path = [self.network_graph.nx_graph.edges[edge_index] for edge_index in np.nonzero(self.x_variable[demand_index, :])]
            ordered_path = nx.dag_longest_path(nx.DiGraph(unordered_path))
            assert (ordered_path[0] == demand_origin) and (ordered_path[-1] == demand_destination)

        # Build latencies
        
        # Check latencies per demand



        self.check_routing()
        self.check_latencies()


class ToyInstance(ProblemInstance):
    def __init__(self) -> None:
        """Toy Instance"""
        pass