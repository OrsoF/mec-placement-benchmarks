import networkx as nx
import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.service_package import ServicePackage
from core.solution import validate_arc_solution


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
        self.validate_components()

        self.number_of_service = self.service_package.number_of_services
        self.demand_per_service = [
            service.task_number for service in self.service_package.services
        ]
        self.total_number_of_demands = sum(self.demand_per_service)

        self.origins = [
            origin
            for service in self.service_package.services
            for origin in service.origin_nodes
        ]
        self.min_flows = [
            flow for service in self.service_package.services for flow in service.min_flows
        ]
        self.max_flows = [
            flow for service in self.service_package.services for flow in service.max_flows
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
            k
            for k, demand_count in enumerate(self.demand_per_service)
            for _ in range(demand_count)
        ]

        self.validate_instance()

    def validate_components(self) -> None:
        assert isinstance(self.network_graph, NetworkGraph)
        assert isinstance(self.service_package, ServicePackage)
        assert self.problem_name

        graph = self.network_graph.nx_graph
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        assert n > 0
        assert m > 0
        assert len(self.network_graph.available_resources) == n
        assert len(self.network_graph.capacities) == m
        assert len(self.network_graph.alphas) == m
        assert len(self.network_graph.betas) == m
        assert all(resource >= 0 for resource in self.network_graph.available_resources)
        assert all(capacity > 0 for capacity in self.network_graph.capacities)
        assert all(alpha >= 0 for alpha in self.network_graph.alphas)
        assert all(beta >= 0 for beta in self.network_graph.betas)

        services = self.service_package.services
        assert services
        assert self.service_package.nx_graph is graph
        assert self.service_package.number_of_services == len(services)
        assert all(
            service.task_number == self.service_package.number_of_demand_per_service
            for service in services
        )
        for service in services:
            assert service.network_graph is self.network_graph
            assert service.task_number == len(service.task_list)
            assert service.needed_resources >= 0
            assert len(service.origin_nodes) == service.task_number
            assert len(service.min_flows) == service.task_number
            assert len(service.max_flows) == service.task_number
            assert len(service.max_latencies) == service.task_number
            assert all(0 <= origin < n for origin in service.origin_nodes)
            assert all(flow >= 0 for flow in service.min_flows)
            assert all(flow >= 0 for flow in service.max_flows)
            assert all(lo <= hi for lo, hi in zip(service.min_flows, service.max_flows))
            assert all(latency >= 0 for latency in service.max_latencies)

    def _set_arc_solution(self, x_variable, y_variable, q_variable):
        x_variable = np.asarray(x_variable)
        y_variable = np.asarray(y_variable)
        q_variable = np.asarray(q_variable)
        assert x_variable.shape == (
            self.total_number_of_demands,
            self.network_graph.nx_graph.number_of_edges(),
        )
        assert y_variable.shape == (
            self.number_of_service,
            self.network_graph.nx_graph.number_of_nodes(),
        )
        assert q_variable.shape == (self.total_number_of_demands,)
        self.x_variable = x_variable
        self.y_variable = y_variable
        self.q_variable = q_variable

    def check_current_solution(self):
        validate_arc_solution(self, self.x_variable, self.y_variable, self.q_variable)

    def validate_instance(self):
        placement_node = int(np.argmax(self.network_graph.available_resources))
        graph = self.network_graph.nx_graph

        x = np.zeros((self.total_number_of_demands, graph.number_of_edges()))
        for d in range(self.total_number_of_demands):
            path = path_node_to_path_tuple(nx.shortest_path(graph, self.origins[d], placement_node))
            for edge in path:
                x[d, self.network_graph.arc_index_dictionary[edge]] = 1.0

        y = np.zeros((self.number_of_service, graph.number_of_nodes()))
        y[:, placement_node] = 1.0
        q = np.asarray(self.min_flows)

        self._set_arc_solution(x, y, q)
        self.check_current_solution()


class ToyInstance(ProblemInstance):
    def __init__(self) -> None:
        pass
