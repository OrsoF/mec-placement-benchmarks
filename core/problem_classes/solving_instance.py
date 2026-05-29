import numpy as np
import networkx as nx

from core.problem_classes.problem_instance import ProblemInstance
from core.graph_utils import (
    path_node_to_path_tuple,
)

rnd_gen = np.random.default_rng(0)


class SolvingInstance:
    """
    Instance of graph that can be updated through methods to solve the original problem. Can test solutions.
    """
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True):
        self.problem_instance = problem_instance
        self.graph = nx.Graph(self.problem_instance.network_graph.nx_graph)
        self.saved_shortest_path = {}
        self.current_solution = None
        self.current_arc_solution = None
        self.initialize_graph_attributes(self.graph)

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
        """Test if the solution is feasible."""

        if (
            len(placement) != self.problem_instance.number_of_service
            or len(routing) != self.problem_instance.total_number_of_demands
            or len(quantities) != self.problem_instance.total_number_of_demands
        ):
            return False

        if not all(0 <= elem < self.graph.number_of_nodes() for elem in placement):
            return False

        if not all(0 <= elem for elem in routing):
            return False

        if not all(
            min_flow <= quantity <= max_flow
            for quantity, min_flow, max_flow in zip(
                quantities,
                self.problem_instance.min_flows,
                self.problem_instance.max_flows,
            )
        ):
            return False

        graph = self.graph.copy()
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
        quantities = list(self.problem_instance.min_flows)

        return (placement, routing, quantities)

    def render_solution(
        self,
        placement: list = None,
        routing: list = None,
        quantities: list = None,
        **plot_kwargs,
    ):
        """Plot a solution using the shared plotting package."""
        from plotting.result_plots import plot_instance_solution

        provided = (placement is not None, routing is not None, quantities is not None)
        if any(provided) and not all(provided):
            raise ValueError("placement, routing, and quantities must be provided together.")

        if not any(provided):
            if self.current_arc_solution is not None:
                x_variable, y_variable, q_variable = self.current_arc_solution
                return plot_instance_solution(
                    self.problem_instance,
                    x=x_variable,
                    y=y_variable,
                    q=q_variable,
                    **plot_kwargs,
                )
            if self.current_solution is None:
                raise ValueError("No solution has been stored or provided.")
            placement, routing, quantities = self.current_solution

        x_variable, y_variable, q_variable = self.solution_to_arc_variables(
            placement, routing, quantities
        )
        return plot_instance_solution(
            self.problem_instance,
            x=x_variable,
            y=y_variable,
            q=q_variable,
            **plot_kwargs,
        )

    def solution_to_arc_variables(
        self, placement: list, routing: list, quantities: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert placement/routing/quantity vectors to plotting-compatible arrays."""
        self._validate_solution_vectors(placement, routing, quantities)

        y_variable = np.zeros(
            (self.problem_instance.number_of_service, self.graph.number_of_nodes()),
            dtype=int,
        )
        for service_index, node in enumerate(placement):
            y_variable[service_index, int(node)] = 1

        x_variable = np.zeros(
            (
                self.problem_instance.total_number_of_demands,
                self.problem_instance.network_graph.nx_graph.number_of_edges(),
            ),
            dtype=int,
        )
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            path = self.compute_nth_shortest_path(
                origin, target, int(routing[demand_index])
            )
            for edge in path_node_to_path_tuple(path):
                edge_index = self.problem_instance.network_graph.arc_index_dictionary[
                    edge
                ]
                x_variable[demand_index, edge_index] = 1

        q_variable = np.asarray(quantities)
        return x_variable, y_variable, q_variable

    def _validate_solution_vectors(
        self, placement: list, routing: list, quantities: list
    ) -> None:
        if len(placement) != self.problem_instance.number_of_service:
            raise ValueError("placement length must match the number of services.")
        if len(routing) != self.problem_instance.total_number_of_demands:
            raise ValueError("routing length must match the number of demands.")
        if len(quantities) != self.problem_instance.total_number_of_demands:
            raise ValueError("quantities length must match the number of demands.")
        if not all(0 <= int(node) < self.graph.number_of_nodes() for node in placement):
            raise ValueError("placement contains an unknown node.")
        if not all(int(path_index) >= 0 for path_index in routing):
            raise ValueError("routing contains a negative path index.")

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

    def _maximize_quantities_greedy(self):
        """Backward-compatible helper used by legacy smoke scripts."""
        placement, routing, quantities = self.basic_placement_and_routing()
        self.current_solution = self.maximize_quantities(
            placement, routing, quantities, addten=False
        )
        self.current_arc_solution = None
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes(self.graph, *self.current_solution)
        return self.current_solution

    def reset_instance(self):
        """Reset mutable graph attributes for compatibility with older scripts."""
        self.graph = self.initialize_graph_attributes(self.graph)
        return self.graph

    def set_variable_solution(self, x_variable, y_variable, q_variable):
        """Store an arc/matrix solution in vector form for rendering."""
        x_variable = np.asarray(x_variable)
        y_variable = np.asarray(y_variable)
        q_variable = np.asarray(q_variable)
        self.problem_instance._set_arc_solution(x_variable, y_variable, q_variable)
        self.current_arc_solution = (x_variable, y_variable, q_variable)
        placement = [
            int(np.argmax(y_variable[service_index]))
            for service_index in range(self.problem_instance.number_of_service)
        ]
        routing = [0 for _ in range(self.problem_instance.total_number_of_demands)]
        quantities = list(q_variable)
        self.current_solution = (placement, routing, quantities)
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes_from_arc_variables(
            self.graph, x_variable, y_variable, q_variable
        )
        return self.current_solution

    def update_graph_attributes_from_arc_variables(
        self,
        graph: nx.Graph,
        x_variable: np.ndarray,
        y_variable: np.ndarray,
        q_variable: np.ndarray,
    ) -> nx.Graph:
        """Update graph attributes from arc/matrix variables."""
        for service_index in range(self.problem_instance.number_of_service):
            target_node = int(np.argmax(y_variable[service_index]))
            required_resources = self.problem_instance.needed_resources[service_index]
            graph.nodes[target_node]["remaining_resources"] -= required_resources

        for demand_index in range(self.problem_instance.total_number_of_demands):
            quantity = q_variable[demand_index]
            for arc_index in np.nonzero(x_variable[demand_index])[0]:
                edge = self.problem_instance.network_graph.arc_list[arc_index]
                graph_edge = edge if graph.has_edge(*edge) else (edge[1], edge[0])
                graph.edges[graph_edge]["remaining_capacity"] -= quantity
                graph.edges[graph_edge]["latency"] += (
                    quantity * graph.edges[graph_edge]["alpha"]
                )

        return graph

    def _get_remaining_quantity_per_demand(
        self, placement: list, routing: list, quantities: list, demand_index: int
    ):
        self._validate_solution_vectors(placement, routing, quantities)

        graph = self.graph.copy()
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
