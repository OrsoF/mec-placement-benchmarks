"""Domain objects for service placement problems."""

import networkx as nx
from numbers import Integral, Real
import numpy as np
from mecbench.domain.validation import validate_arc_solution


def check_network_graph(nx_graph: nx.DiGraph):
    """
    Checks if the graph is a DiGraph and is symmetric.
    """
    assert isinstance(
        nx_graph, nx.DiGraph
    ), "Considered graph is not a DiGraph instance."
    assert is_graph_symmetric(nx_graph), "Considered graph is not symmetric."


def is_graph_symmetric(nx_graph: nx.DiGraph) -> bool:
    """
    Check if a given DiGraph is symmetric.
    """
    for edge in nx_graph.edges:
        if (edge[1], edge[0]) not in nx_graph.edges:
            return False
    return True

class NetworkGraph:
    def __init__(
        self,
        base_graph: nx.Graph,
        node_resources: list[float],
        edge_capacity: list[float],
        latency_alpha_factor: list[float],
        latency_beta_factor: list[float],
    ):
        # The graph needs to be symmetric. We label nodes with integers.
        if not isinstance(base_graph, nx.DiGraph):
            self.nx_graph = nx.DiGraph(base_graph)
            self.nx_graph = nx.relabel.relabel_nodes(
                self.nx_graph, {node: i for i, node in enumerate(self.nx_graph.nodes())}
            )
            self.make_graph_symmetric()
            self.remove_all_attributes()
        else:
            self.nx_graph = base_graph
            self.make_graph_symmetric()

        check_network_graph(self.nx_graph)

        self.set_node_attribute(node_resources, "available_resources")
        self.set_edge_attribute(edge_capacity, "capacity")
        self.set_edge_attribute(latency_alpha_factor, "alpha")
        self.set_edge_attribute(latency_beta_factor, "beta")

        self.alphas: list[float] = [
            self.nx_graph.edges[edge]["alpha"] for edge in self.nx_graph.edges
        ]
        self.betas: list[float] = [
            self.nx_graph.edges[edge]["beta"] for edge in self.nx_graph.edges
        ]
        self.capacities = [
            self.nx_graph.edges[edge]["capacity"] for edge in self.nx_graph.edges
        ]
        self.available_resources = [
            self.nx_graph.nodes[node]["available_resources"]
            for node in self.nx_graph.nodes
        ]
        self.arc_index_dictionary = {
            arc: i for i, arc in enumerate(self.nx_graph.edges)
        }
        self.arc_list = list(self.nx_graph.edges)

    def set_node_attribute(self, attributes: list[float], name: str):
        assert (
            len(attributes) == self.nx_graph.number_of_nodes()
        ), "There not as many node parameters as the number of nodes in the graph."
        attribute_map = {
            node_index: attributes[node_index]
            for node_index in range(self.nx_graph.number_of_nodes())
        }
        nx.classes.function.set_node_attributes(self.nx_graph, attribute_map, name)

    def set_edge_attribute(self, attributes: list[float], name: str):
        edges = list(self.nx_graph.edges)

        if len(attributes) == len(edges) // 2:
            undirected_edges = []
            seen = set()
            for u, v in edges:
                if (v, u) in seen:
                    continue
                undirected_edges.append((u, v))
                seen.add((u, v))

            assert len(attributes) == len(
                undirected_edges
            ), "{} parameters not equal to {} undirected edges in graph.".format(
                len(attributes), len(undirected_edges)
            )
            attribute_map = {}
            for edge, value in zip(undirected_edges, attributes):
                u, v = edge
                attribute_map[(u, v)] = value
                attribute_map[(v, u)] = value
        else:
            assert len(attributes) == len(
                edges
            ), "{} parameters not equal to {} edges in graph.".format(
                len(attributes), len(edges)
            )
            attribute_map = {
                edge: attributes[edge_index] for edge_index, edge in enumerate(edges)
            }
        nx.classes.function.set_edge_attributes(self.nx_graph, attribute_map, name)

    def make_graph_symmetric(self):
        for arc in self.nx_graph.edges:
            if (arc[1], arc[0]) not in self.nx_graph.edges:
                self.nx_graph.add_edge(arc[1], arc[0])

    def remove_all_attributes(self):
        for node in self.nx_graph.nodes:
            keys = list(self.nx_graph.nodes[node].keys())
            for key in keys:
                del self.nx_graph.nodes[node][key]
        for edge in self.nx_graph.edges:
            keys = list(self.nx_graph.edges[edge].keys())
            for key in keys:
                del self.nx_graph.edges[edge][key]
        return self.nx_graph
    
    def draw_graph(self):
        nx.draw(self.nx_graph, with_labels = True)


class Demand:
    """
    A demand is a certain amount of resources needed at a destination
    node and a flow in [flow_min, flow_max] that need to transit.
    """

    def __init__(
        self,
        origin_node_index: int,
        flow_min: float,
        flow_max: float,
        max_latency: float,
    ):
        if not isinstance(origin_node_index, Integral) or isinstance(
            origin_node_index, bool
        ):
            raise TypeError("origin_node_index must be an int.")
        if not isinstance(flow_min, Real):
            raise TypeError("flow_min must be numeric.")
        if not isinstance(flow_max, Real):
            raise TypeError("flow_max must be numeric.")
        if not isinstance(max_latency, Real):
            raise TypeError("max_latency must be numeric.")
        if origin_node_index < 0:
            raise ValueError("origin_node_index must be non-negative.")
        if flow_min < 0:
            raise ValueError("flow_min must be non-negative.")
        if flow_max < 0:
            raise ValueError("flow_max must be non-negative.")
        if flow_min > flow_max:
            raise ValueError("flow_min must be <= flow_max.")
        if max_latency < 0:
            raise ValueError("max_latency must be non-negative.")

        self.origin_node_index = origin_node_index
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.max_latency = max_latency


class Service:
    def __init__(
        self,
        task_list: list[Demand],
        network_graph: NetworkGraph,
        resources_needed: float,
    ):
        if not isinstance(network_graph, NetworkGraph):
            raise TypeError("network_graph must be a NetworkGraph.")
        if resources_needed < 0:
            raise ValueError("resources_needed must be non-negative.")
        if not task_list:
            raise ValueError("task_list must not be empty.")
        if not all(isinstance(task, Demand) for task in task_list):
            raise TypeError("task_list must contain only Demand instances.")

        self.network_graph = network_graph
        self.node_number = self.network_graph.nx_graph.number_of_nodes()
        self.edge_number = self.network_graph.nx_graph.number_of_edges()
        self.needed_resources = resources_needed

        self.task_number = 0
        self.task_list = []

        self.origin_nodes = []
        self.min_flows = []
        self.max_flows = []
        self.max_latencies = []

        for task in task_list:
            self.add_task(task)

    def add_task(self, task_element: Demand):
        self._check_task(task_element)

        self.current_task_element = task_element
        self.task_list.append(self.current_task_element)
        self.task_number += 1

        self.origin_nodes.append(self.current_task_element.origin_node_index)
        self.min_flows.append(self.current_task_element.flow_min)
        self.max_flows.append(self.current_task_element.flow_max)
        self.max_latencies.append(self.current_task_element.max_latency)

    def _check_task(self, task: Demand):
        if not 0 <= task.origin_node_index < self.node_number:
            raise ValueError(
                f"Demand origin node {task.origin_node_index} is outside graph nodes."
            )


class ServicePackage:
    """A list of Services."""

    def __init__(self, services: list[Service], instance_name: str) -> None:
        if not services:
            raise ValueError("services must not be empty.")
        if not instance_name:
            raise ValueError("instance_name must not be empty.")
        if not all(isinstance(service, Service) for service in services):
            raise TypeError("services must contain only Service instances.")

        first_network_graph = services[0].network_graph
        first_task_number = services[0].task_number
        for service in services:
            if service.network_graph is not first_network_graph:
                raise ValueError("All services must use the same NetworkGraph instance.")
            if service.task_number != first_task_number:
                raise ValueError(
                    "All services must have the same number of demands."
                )

        self.nx_graph: nx.Graph = services[0].network_graph.nx_graph

        self.services: list[Service] = []
        for service in services:
            self.services.append(service)

        self.instance_name = instance_name

        self.number_of_services: int = len(services)
        self.number_of_demand_per_service: int = services[0].task_number
        self.number_of_nodes = self.nx_graph.number_of_nodes()
        self.number_of_edges = self.nx_graph.number_of_edges()


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

        # A baseline placement is a candidate, not a proof of instance feasibility.
        self._initialize_baseline()

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
        """Check the historical baseline, without claiming global infeasibility."""
        self._initialize_baseline()
        self.check_current_solution()

    def _initialize_baseline(self):
        placement_node = int(np.argmax(self.network_graph.available_resources))
        graph = self.network_graph.nx_graph

        x = np.zeros((self.total_number_of_demands, graph.number_of_edges()))
        for d in range(self.total_number_of_demands):
            nodes = nx.shortest_path(graph, self.origins[d], placement_node)
            path = list(zip(nodes, nodes[1:]))
            for edge in path:
                x[d, self.network_graph.arc_index_dictionary[edge]] = 1.0

        y = np.zeros((self.number_of_service, graph.number_of_nodes()))
        y[:, placement_node] = 1.0
        q = np.asarray(self.min_flows)

        self._set_arc_solution(x, y, q)


class ToyInstance(ProblemInstance):
    def __init__(self) -> None:
        pass
