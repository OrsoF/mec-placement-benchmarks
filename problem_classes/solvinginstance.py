import numpy as np
import networkx as nx

from problem_classes.probleminstance import ProblemInstance
from utils.graph_utils import (
    path_node_to_path_tuple,
    edges_to_path,
)

rnd_gen = np.random.default_rng(0)


class SolvingInstance:
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True):
        self.problem_instance = problem_instance
        self.graph = nx.Graph(self.problem_instance.network_graph.nx_graph)
        self.saved_shortest_path = {}

    def initialize_graph_attributes(self, graph: nx.Graph) -> nx.Graph:
        """Set graph attributes the first time."""
        for node in graph.nodes:
            graph.nodes[node]["remaining_resources"] = graph.nodes[node][
                "available_resources"
            ]

        for edge in graph.edges:
            graph.edges[edge]["remaining_capacity"] = graph.edges[edge]["capacity"]
            graph.edges[edge]["latency"] = graph.edges[edge]["beta"]

        return graph

    def update_graph_attributes(
        self, graph: nx.Graph, placement: list, routing: list, quantities: list
    ) -> nx.Graph:
        """Update the graph attributes according to the variables."""
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            path = self.compute_nth_shortest_path(origin, target, routing[demand_index])
            demand_path = path_node_to_path_tuple(path)

            for edge in demand_path:
                graph.edges[edge]["remaining_capacity"] -= quantities[demand_index]
                graph.edges[edge]["latency"] += (
                    quantities[demand_index] * graph.edges[edge]["alpha"]
                )

        for service_index in range(self.problem_instance.number_of_service):
            target_node = placement[service_index]
            required_resources = self.problem_instance.needed_resources[service_index]
            graph.nodes[target_node]["remaining_resources"] -= required_resources

        return graph

    def validate_graph_attributes(
        self, graph: nx.Graph, placement: list, routing: list, quantities: list
    ) -> bool:
        """Check if the graph attributes checks constraints."""
        for edge in graph.edges:  # Capacité pour chaque arc
            if graph.edges[edge]["remaining_capacity"] < 0.0:  # Edge capacity
                return False

        for demand_index in range(
            self.problem_instance.total_number_of_demands
        ):  # Pour chaque demande, quantité et latence limite
            if (
                quantities[demand_index] > self.problem_instance.max_flows[demand_index]
            ):  # Max flows
                return False

            service_index = self.problem_instance.demand_to_service[
                demand_index
            ]  # Latency
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            demand_path = path_node_to_path_tuple(
                self.compute_nth_shortest_path(origin, target, routing[demand_index])
            )
            latency = sum(graph.edges[edge]["latency"] for edge in demand_path)
            if latency > self.problem_instance.latencies[demand_index]:
                return False

        for node in graph.nodes:  # Ressource pour chaque noeud
            if graph.nodes[node]["remaining_resources"] < 0.0:
                return False

        return True

    def test_solution(self, placement: list, routing: list, quantities: list) -> bool:
        # Vérifier les inputs ici

        if not all(0 <= elem < self.graph.number_of_nodes() for elem in placement):
            return False

        if not all(0 <= elem for elem in routing):
            return False

        if not all(
            self.problem_instance.min_flows[0]
            <= elem
            <= self.problem_instance.max_flows[0]
            for elem in quantities
        ):
            return False

        graph = self.graph  # .copy()
        graph = self.initialize_graph_attributes(graph)
        graph = self.update_graph_attributes(graph, placement, routing, quantities)
        return self.validate_graph_attributes(graph, placement, routing, quantities)

    def basic_placement_and_routing(self) -> tuple:
        """All services in the cloud, shortest paths and smallest quantities."""
        cloud_node = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )

        placement = [cloud_node for _ in range(self.problem_instance.number_of_service)]
        routing = [0 for _ in range(self.problem_instance.total_number_of_demands)]
        quantities = [1 for _ in range(self.problem_instance.total_number_of_demands)]

        return (placement, routing, quantities)

    def render_solution(self, placement: list, routing: list, quantities: list):

        for service_index in range(self.problem_instance.number_of_service):
            print(f"Service {service_index} placed on node {placement[service_index]}.")
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
            print("Path for demand {} : {}".format(demand_index, routing[demand_index]))
        print()

        for demand_index in range(self.problem_instance.total_number_of_demands):
            print(
                "Quantity for demand {} : {} / {}".format(
                    demand_index,
                    quantities[demand_index],
                    self.problem_instance.max_flows[demand_index],
                )
            )
        print()

        for demand_index in range(self.problem_instance.total_number_of_demands):
            path_index = routing[demand_index]
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            path_node = self.compute_nth_shortest_path(origin, target, path_index)
            path_edge = path_node_to_path_tuple(path_node)

            latency = self.get_latency_from_graph(path_edge)
            print(
                "Latency for demand {} : {} / {}".format(
                    demand_index, latency, self.problem_instance.latencies[demand_index]
                )
            )
        print()
        print("Solution reached : {}".format(sum(quantities)))

        for edge in self.graph.edges:
            print("Alpha : ", self.graph.edges[edge]["alpha"])
            print("Beta : ", self.graph.edges[edge]["beta"])
            print("Latency : ", self.graph.edges[edge]["latency"])
        print()

    def get_latency_from_graph(self, path_edge: list) -> float:
        return sum(self.graph.edges[edge]["latency"] for edge in path_edge)

    def maximize_quantities(self, placement, routing, quantities, addten=True):
        assert self.test_solution(placement, routing, quantities)

        step = (
            10
            * self.problem_instance.total_number_of_demands
            * self.problem_instance.max_flows[0]
        )

        for _ in range(step):
            new_quantites = quantities.copy()
            concerned_demand = rnd_gen.integers(
                self.problem_instance.total_number_of_demands
            )

            add = 10 if addten else 1
            new_quantites[concerned_demand] += add
            if self.test_solution(placement, routing, new_quantites):
                quantities = new_quantites

        return placement, routing, quantities

    def _get_remaining_quantity_per_demand(
        self, placement: list, routing: list, quantities: list, demand_index: int
    ):
        assert self._are_variable_correct(
            placement, routing, quantities
        ), "The current placement and routing are not correct."

        graph = self.initialize_graph_attributes(graph)
        graph = self.update_graph_attributes(graph, placement, routing, quantities)

        service_index = self.problem_instance.demand_to_service[demand_index]
        origin = self.problem_instance.origins[demand_index]
        target = placement[service_index]
        path = self.compute_nth_shortest_path(origin, target, routing[demand_index])
        if len(path) == 1:
            return (
                self.problem_instance.max_flows[demand_index] - quantities[demand_index]
            )

        demand_path = path_node_to_path_tuple(path)

        # On peut ajouter des trucs sur la latence
        remaining_latency = self.problem_instance.latencies[demand_index] - sum(
            graph.edges[edge]["latency"] for edge in demand_path
        )
        remaining_quantity_latency = remaining_latency / min(
            graph.edges[edge]["alpha"] for edge in demand_path
        )

        # On peut ajouter des trucs sur les capacités
        remaining_quantity_capacity = min(
            graph.edges[edge]["remaining_capacity"] for edge in demand_path
        )

        # On peut ajouter des trucs sur le flow max
        remaining_quantity_flow_max = (
            self.problem_instance.max_flows[demand_index] - quantities[demand_index]
        )

        remaining_quantity_to_add = min(
            remaining_quantity_latency,
            remaining_quantity_capacity,
            remaining_quantity_flow_max,
        )
        assert remaining_quantity_to_add >= 0

        return np.round(remaining_quantity_to_add)

    def compute_nth_shortest_path(
        self, source: int, destination: int, index: int
    ) -> list:
        """
        Returns the index-th shortest path from source to destination in the form [2, 1, 3].
        """
        index = int(index)

        if (source, destination, index) in self.saved_shortest_path:
            return self.saved_shortest_path[(source, destination, index)]

        for i, path in enumerate(
            nx.algorithms.simple_paths.shortest_simple_paths(
                self.graph,
                source,
                destination,
            )
        ):
            last_visited_path = path
            if i == index:
                break

        self.saved_shortest_path[(source, destination, index)] = last_visited_path
        return last_visited_path

    def test_basic_placement(self, print_solution=False):
        placement, routing, quantities = self.basic_placement_and_routing()
        it_works = self.test_solution(placement, routing, quantities)
        if it_works:
            print("Basic placement and routing works.")

        if print_solution:
            self.render_solution(placement, routing, quantities)

        assert it_works, "Basic placement and routing does not work."
