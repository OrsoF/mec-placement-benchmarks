"""Mutable graphs, candidate state and routing caches owned by one solver."""

import networkx as nx
import numpy as np

from mecbench.domain.conversions import compact_to_arc, validate_compact_shapes
from mecbench.domain.loads import residual_arrays
from mecbench.domain.network import NetworkSnapshot
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import problem_arrays
from mecbench.domain.solution import Solution
from mecbench.domain.validation import validate_arc_solution
from mecbench.utils.heuristic_solvers import basic_solution, remaining_quantity
from mecbench.utils.solver_paths import ShortestPathCache


def apply_residuals_to_graph(problem, graph: nx.Graph, x, y, q) -> nx.Graph:
    residual = residual_arrays(problem, x, y, q)
    for u in graph.nodes:
        graph.nodes[u]["remaining_resources"] = residual["remaining_resources"][u]

    for a, e in enumerate(problem_arrays(problem).arcs):
        if not graph.has_edge(*e):
            continue
        graph.edges[e]["remaining_capacity"] = residual["remaining_capacity"][a]
        graph.edges[e]["latency"] = residual["arc_latency"][a]
    return graph


class SolverWorkspace:
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True, *, seed: int = 0):
        self.problem_instance = problem_instance
        self.data = problem_arrays(problem_instance)
        self.graph = NetworkSnapshot.make_graph(
            self.data, self.problem_instance.network_graph.nx_graph
        )
        self.path_cache = ShortestPathCache(self.graph)
        self.saved_shortest_path = self.path_cache._paths
        self.rng = np.random.default_rng(seed)
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
        return apply_residuals_to_graph(self.data, graph, x, y, q)

    def validate_graph_attributes(self, graph, placement, routing, quantities) -> bool:
        return self.test_compact_solution(placement, routing, quantities)

    def test_solution(self, placement, routing, quantities) -> bool:
        return self.test_compact_solution(placement, routing, quantities)

    def basic_placement_and_routing(self) -> tuple:
        return basic_solution(self.data, self.graph)

    def render_solution(
        self,
        placement: list = None,
        routing: list = None,
        quantities: list = None,
        **plot_kwargs,
    ):
        from mecbench.benchmarks.reporting import plot_instance_solution

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
            self.data,
            self.graph,
            self.compute_nth_shortest_path,
            placement,
            routing,
            quantities,
        )

    def path_edge_indices(self, source: int, destination: int, index: int) -> np.ndarray:
        path = self.compute_nth_shortest_path(source, destination, index)
        arc_index = self.data.arc_index
        return np.fromiter(
            (arc_index[(path[i], path[i + 1])] for i in range(len(path) - 1)),
            dtype=int,
            count=max(0, len(path) - 1),
        )

    def compact_path_indices(self, placement, routing) -> list[np.ndarray]:
        return [
            self.path_edge_indices(
                self.data.origins[d],
                int(placement[self.data.demand_to_service[d]]),
                int(routing[d]),
            )
            for d in range(self.data.n_demands)
        ]

    def compact_loads(self, placement, routing, quantities):
        paths = self.compact_path_indices(placement, routing)
        q = np.asarray(quantities, dtype=float)
        edge_load = np.zeros(self.data.n_arcs)
        for d, idx in enumerate(paths):
            edge_load[idx] += q[d]

        node_load = np.bincount(
            np.asarray(placement, dtype=int),
            weights=np.asarray(self.data.service_resources, dtype=float),
            minlength=self.data.n_nodes,
        )
        return paths, edge_load, node_load

    def test_compact_solution(self, placement, routing, quantities) -> bool:
        try:
            x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
            validate_arc_solution(self.data, x, y, q)
        except (IndexError, ValueError, TypeError, KeyError, nx.NetworkXException):
            return False
        return True

    def _validate_solution_vectors(self, placement, routing, quantities) -> None:
        validate_compact_shapes(self.data, self.graph, placement, routing, quantities)

    def get_latency_from_graph(self, path_edge: list) -> float:
        return sum(self.graph.edges[e]["latency"] for e in path_edge)

    def maximize_quantities(self, placement, routing, quantities, addten=True):
        assert self.test_solution(placement, routing, quantities)
        step = 10 * self.data.n_demands * self.data.max_flows[0]
        q = list(quantities)
        for _ in range(int(step)):
            q_new = q.copy()
            d = self.rng.integers(self.data.n_demands)
            q_new[d] += 10 if addten else 1
            if self.test_solution(placement, routing, q_new):
                q = q_new
        return placement, routing, q

    def _maximize_quantities_greedy(self):
        placement, routing, quantities = self.basic_placement_and_routing()
        self.current_solution = self.maximize_quantities(
            placement, routing, quantities, addten=False
        )
        self.current_arc_solution = None
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes(self.graph, *self.current_solution)
        return self.current_solution

    def reset_instance(self):
        self.graph = self.initialize_graph_attributes(self.graph)
        return self.graph

    def set_variable_solution(self, x_variable, y_variable, q_variable):
        solution = Solution(x_variable, y_variable, q_variable)
        validate_arc_solution(self.data, solution.x, solution.y, solution.q)
        x_variable, y_variable, q_variable = solution.x, solution.y, solution.q
        self.current_arc_solution = (x_variable, y_variable, q_variable)
        placement = [int(np.argmax(y_variable[k])) for k in range(self.data.n_services)]
        routing = [0] * self.data.n_demands
        quantities = list(q_variable)
        self.current_solution = (placement, routing, quantities)
        self.graph = self.initialize_graph_attributes(self.graph)
        self.graph = self.update_graph_attributes_from_arc_variables(
            self.graph, x_variable, y_variable, q_variable
        )
        return self.current_solution

    def update_graph_attributes_from_arc_variables(self, graph, x_variable, y_variable, q_variable):
        return apply_residuals_to_graph(self.data, graph, x_variable, y_variable, q_variable)

    def _get_remaining_quantity_per_demand(self, placement, routing, quantities, demand_index):
        x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
        return remaining_quantity(self.data, x, y, q, demand_index)

    def compute_nth_shortest_path(self, source: int, destination: int, index: int) -> list:
        return self.path_cache.nth_shortest_path(int(source), int(destination), int(index))

    def test_basic_placement(self, print_solution=False):
        placement, routing, quantities = self.basic_placement_and_routing()
        works = self.test_solution(placement, routing, quantities)
        if works:
            print("Basic placement and routing works.")
        if print_solution:
            self.render_solution(placement, routing, quantities)
        assert works, "Basic placement and routing does not work."


SolvingInstance = SolverWorkspace
