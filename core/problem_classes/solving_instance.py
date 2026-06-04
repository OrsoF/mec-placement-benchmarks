import networkx as nx
import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.problem_instance import ProblemInstance
from core.solution import (
    apply_residuals_to_graph,
    basic_solution,
    compact_to_arc,
    validate_compact_shapes,
    validate_arc_solution,
)

rnd_gen = np.random.default_rng(0)


class SolvingInstance:
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True):
        self.problem_instance = problem_instance
        self.graph = nx.Graph(self.problem_instance.network_graph.nx_graph)
        self.saved_shortest_path = {}
        self.current_solution = None
        self.current_arc_solution = None
        self.initialize_graph_attributes(self.graph)

    def initialize_graph_attributes(self, graph: nx.Graph) -> nx.Graph:
        for v in graph.nodes:
            graph.nodes[v]["remaining_resources"] = graph.nodes[v]["available_resources"]
        for e in graph.edges:
            graph.edges[e]["remaining_capacity"] = graph.edges[e]["capacity"]
            graph.edges[e]["latency"] = graph.edges[e]["beta"]
        return graph

    def update_graph_attributes(self, graph, placement, routing, quantities) -> nx.Graph:
        x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
        return apply_residuals_to_graph(self.problem_instance, graph, x, y, q)

    def validate_graph_attributes(self, graph, placement, routing, quantities) -> bool:
        try:
            x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
            validate_arc_solution(self.problem_instance, x, y, q)
        except (AssertionError, IndexError, ValueError, nx.NetworkXException):
            return False
        return True

    def test_solution(self, placement, routing, quantities) -> bool:
        try:
            x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
            validate_arc_solution(self.problem_instance, x, y, q)
        except (AssertionError, IndexError, ValueError, nx.NetworkXException):
            return False
        return True

    def basic_placement_and_routing(self) -> tuple:
        return basic_solution(self.problem_instance, self.graph)

    def render_solution(
        self,
        placement: list = None,
        routing: list = None,
        quantities: list = None,
        **plot_kwargs,
    ):
        from plotting.result_plots import plot_instance_solution

        provided = (placement is not None, routing is not None, quantities is not None)
        if any(provided) and not all(provided):
            raise ValueError("placement, routing, and quantities must be provided together.")

        if not any(provided):
            if self.current_arc_solution is not None:
                return plot_instance_solution(
                    self.problem_instance,
                    *self.current_arc_solution,
                    **plot_kwargs,
                )
            if self.current_solution is None:
                raise ValueError("No solution has been stored or provided.")
            placement, routing, quantities = self.current_solution

        return plot_instance_solution(
            self.problem_instance,
            *self.solution_to_arc_variables(placement, routing, quantities),
            **plot_kwargs,
        )

    def solution_to_arc_variables(self, placement, routing, quantities):
        return compact_to_arc(
            self.problem_instance,
            self.problem_instance.network_graph.nx_graph,
            self.compute_nth_shortest_path,
            placement,
            routing,
            quantities,
        )

    def _validate_solution_vectors(self, placement, routing, quantities) -> None:
        validate_compact_shapes(self.problem_instance, self.graph, placement, routing, quantities)

    def get_latency_from_graph(self, path_edge: list) -> float:
        return sum(self.graph.edges[e]["latency"] for e in path_edge)

    def maximize_quantities(self, placement, routing, quantities, addten=True):
        assert self.test_solution(placement, routing, quantities)
        step = 10 * self.problem_instance.total_number_of_demands * self.problem_instance.max_flows[0]
        q = list(quantities)
        for _ in range(step):
            q_new = q.copy()
            d = rnd_gen.integers(self.problem_instance.total_number_of_demands)
            q_new[d] += 10 if addten else 1
            if self.test_solution(placement, routing, q_new):
                q = q_new
        return placement, routing, q

    def _maximize_quantities_greedy(self):
        placement, routing, quantities = self.basic_placement_and_routing()
        self.current_solution = self.maximize_quantities(placement, routing, quantities, addten=False)
        self.current_arc_solution = None
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes(self.graph, *self.current_solution)
        return self.current_solution

    def reset_instance(self):
        self.graph = self.initialize_graph_attributes(self.graph)
        return self.graph

    def set_variable_solution(self, x_variable, y_variable, q_variable):
        x_variable = np.asarray(x_variable)
        y_variable = np.asarray(y_variable)
        q_variable = np.asarray(q_variable)
        self.problem_instance._set_arc_solution(x_variable, y_variable, q_variable)
        self.current_arc_solution = (x_variable, y_variable, q_variable)
        placement = [int(np.argmax(y_variable[k])) for k in range(self.problem_instance.number_of_service)]
        routing = [0] * self.problem_instance.total_number_of_demands
        quantities = list(q_variable)
        self.current_solution = (placement, routing, quantities)
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes_from_arc_variables(
            self.graph, x_variable, y_variable, q_variable
        )
        return self.current_solution

    def update_graph_attributes_from_arc_variables(self, graph, x_variable, y_variable, q_variable):
        return apply_residuals_to_graph(
            self.problem_instance, graph, x_variable, y_variable, q_variable
        )

    def _get_remaining_quantity_per_demand(self, placement, routing, quantities, demand_index):
        self._validate_solution_vectors(placement, routing, quantities)
        graph = self.graph.copy()
        self.initialize_graph_attributes(graph)
        self.update_graph_attributes(graph, placement, routing, quantities)

        k = self.problem_instance.demand_to_service[demand_index]
        path = self.compute_nth_shortest_path(
            self.problem_instance.origins[demand_index],
            placement[k],
            routing[demand_index],
        )
        if len(path) == 1:
            return self.problem_instance.max_flows[demand_index] - quantities[demand_index]

        demand_path = path_node_to_path_tuple(path)
        remaining_latency = self.problem_instance.latencies[demand_index] - sum(
            graph.edges[e]["latency"] for e in demand_path
        )
        latency_room = remaining_latency / min(graph.edges[e]["alpha"] for e in demand_path)
        capacity_room = min(graph.edges[e]["remaining_capacity"] for e in demand_path)
        flow_room = self.problem_instance.max_flows[demand_index] - quantities[demand_index]
        return np.round(min(latency_room, capacity_room, flow_room))

    def compute_nth_shortest_path(self, source: int, destination: int, index: int) -> list:
        index = int(index)
        key = (int(source), int(destination), index)
        if key in self.saved_shortest_path:
            return self.saved_shortest_path[key]

        last_path = None
        for i, path in enumerate(
            nx.algorithms.simple_paths.shortest_simple_paths(self.graph, source, destination)
        ):
            last_path = path
            if i == index:
                break
        if last_path is None:
            raise nx.NetworkXNoPath(f"No path from {source} to {destination}.")
        if i != index:
            raise IndexError(index)
        self.saved_shortest_path[key] = last_path
        return last_path

    def test_basic_placement(self, print_solution=False):
        placement, routing, quantities = self.basic_placement_and_routing()
        works = self.test_solution(placement, routing, quantities)
        if works:
            print("Basic placement and routing works.")
        if print_solution:
            self.render_solution(placement, routing, quantities)
        assert works, "Basic placement and routing does not work."
