"""Solution conversion, residuals, and compact-solution adapter."""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from mecbench.domain.graph_utils import path_node_to_path_tuple
from mecbench.domain.validation import validate_arc_solution
from mecbench.domain.problem import ProblemInstance


@dataclass(frozen=True, eq=False)
class Solution:
    """Owned, read-only arc routing, service placement and demand quantities.

    Columns follow ``problem.network_graph.arc_list`` and integer node indices.
    Construction copies the arrays; feasibility is checked by ``validate``.
    """

    x: np.ndarray
    y: np.ndarray
    q: np.ndarray

    def __post_init__(self):
        for name in ("x", "y", "q"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def to_int_array(x) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=float)).astype(int)


def split_state(problem, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(s)
    S = problem.number_of_service
    D = problem.total_number_of_demands
    return s[:S], s[S : S + D], s[S + D : S + 2 * D]


def make_state(placement, routing, quantities) -> np.ndarray:
    return to_int_array(np.concatenate((placement, routing, quantities)))


def basic_solution(problem, graph: nx.Graph) -> tuple[list[int], list[int], list[int]]:
    resources = np.fromiter(
        (graph.nodes[v]["available_resources"] for v in graph.nodes),
        dtype=float,
        count=graph.number_of_nodes(),
    )
    cloud = int(np.argmax(resources))
    return (
        [cloud] * problem.number_of_service,
        [0] * problem.total_number_of_demands,
        list(problem.min_flows),
    )


def basic_state(problem, graph: nx.Graph) -> np.ndarray:
    return make_state(*basic_solution(problem, graph))


def clip_solution(problem, graph: nx.Graph, max_path, placement, routing, quantities):
    n = graph.number_of_nodes()
    p = np.clip(placement, 0, n - 1).astype(int).tolist()
    r = np.clip(routing, 0, max_path - 1).astype(int).tolist()
    q = np.clip(quantities, problem.min_flows, problem.max_flows).astype(int).tolist()
    return p, r, q


def clip_state(problem, graph: nx.Graph, max_path, s) -> np.ndarray:
    s = to_int_array(s)
    p, r, q = split_state(problem, s)
    p[:], r[:], q[:] = clip_solution(problem, graph, max_path, p, r, q)
    return s


def greedy_quantity_lift(problem, checker, rng, placement, routing, quantities, steps, increment):
    q = list(quantities)
    D = problem.total_number_of_demands
    for _ in range(steps):
        d = int(rng.integers(D))
        if q[d] >= problem.max_flows[d]:
            continue
        delta = min(increment, problem.max_flows[d] - q[d])
        q[d] += delta
        if not checker.test_solution(placement, routing, q):
            q[d] -= delta
    return placement, routing, q


def repair_solution(problem, graph, checker, max_path, placement, routing, quantities):
    p, r, q = clip_solution(problem, graph, max_path, placement, routing, quantities)
    if checker.test_solution(p, r, q):
        return p, r, q
    q = list(problem.min_flows)
    if checker.test_solution(p, r, q):
        return p, r, q
    return None


def validate_compact_shapes(problem, graph: nx.Graph, placement, routing, quantities):
    vectors = (
        ("placement", placement, problem.number_of_service),
        ("routing", routing, problem.total_number_of_demands),
        ("quantities", quantities, problem.total_number_of_demands),
    )
    for name, values, size in vectors:
        values = np.asarray(values, dtype=float)
        if values.shape != (size,):
            raise ValueError(f"{name} shape must be ({size},).")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if name != "quantities" and np.any(values != np.rint(values)):
            raise ValueError(f"{name} must contain integer indices.")
    placement = np.asarray(placement, dtype=float)
    routing = np.asarray(routing, dtype=float)
    if np.any(placement < 0) or np.any(placement >= graph.number_of_nodes()):
        raise ValueError("placement contains an unknown node.")
    if np.any(routing < 0):
        raise ValueError("routing contains a negative path index.")


def compact_to_arc(problem, graph: nx.Graph, path_fn, placement, routing, quantities):
    validate_compact_shapes(problem, graph, placement, routing, quantities)

    y = np.zeros((problem.number_of_service, graph.number_of_nodes()), dtype=float)
    y[np.arange(problem.number_of_service), np.asarray(placement, dtype=int)] = 1.0

    x = np.zeros((problem.total_number_of_demands, graph.number_of_edges()), dtype=float)
    for d in range(problem.total_number_of_demands):
        k = problem.demand_to_service[d]
        path = path_fn(problem.origins[d], int(placement[k]), int(routing[d]))
        for e in path_node_to_path_tuple(path):
            x[d, problem.network_graph.arc_index_dictionary[e]] = 1.0

    return x, y, np.asarray(quantities, dtype=float)


def residual_arrays(problem, x, y, q):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    edge_load = x.T @ q
    node_load = y.T @ np.asarray(problem.needed_resources, dtype=float)
    capacities = np.asarray(problem.network_graph.capacities, dtype=float)
    resources = np.asarray(problem.network_graph.available_resources, dtype=float)
    return {
        "edge_load": edge_load,
        "node_load": node_load,
        "remaining_capacity": capacities - edge_load,
        "remaining_resources": resources - node_load,
    }


def apply_residuals_to_graph(problem, graph: nx.Graph, x, y, q) -> nx.Graph:
    residual = residual_arrays(problem, x, y, q)
    for u in graph.nodes:
        graph.nodes[u]["remaining_resources"] = residual["remaining_resources"][u]

    for a, e in enumerate(problem.network_graph.arc_list):
        if not graph.has_edge(*e):
            continue
        load = residual["edge_load"][a]
        graph.edges[e]["remaining_capacity"] = residual["remaining_capacity"][a]
        graph.edges[e]["latency"] = graph.edges[e]["beta"] + load * graph.edges[e]["alpha"]
    return graph


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
        return self.test_compact_solution(placement, routing, quantities)

    def test_solution(self, placement, routing, quantities) -> bool:
        return self.test_compact_solution(placement, routing, quantities)

    def basic_placement_and_routing(self) -> tuple:
        return basic_solution(self.problem_instance, self.graph)

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
            self.problem_instance,
            self.problem_instance.network_graph.nx_graph,
            self.compute_nth_shortest_path,
            placement,
            routing,
            quantities,
        )

    def path_edge_indices(self, source: int, destination: int, index: int) -> np.ndarray:
        path = self.compute_nth_shortest_path(source, destination, index)
        arc_index = self.problem_instance.network_graph.arc_index_dictionary
        return np.fromiter(
            (arc_index[(path[i], path[i + 1])] for i in range(len(path) - 1)),
            dtype=int,
            count=max(0, len(path) - 1),
        )

    def compact_path_indices(self, placement, routing) -> list[np.ndarray]:
        problem = self.problem_instance
        return [
            self.path_edge_indices(
                problem.origins[d],
                int(placement[problem.demand_to_service[d]]),
                int(routing[d]),
            )
            for d in range(problem.total_number_of_demands)
        ]

    def compact_loads(self, placement, routing, quantities):
        problem = self.problem_instance
        paths = self.compact_path_indices(placement, routing)
        q = np.asarray(quantities, dtype=float)
        edge_load = np.zeros(problem.network_graph.nx_graph.number_of_edges())
        for d, idx in enumerate(paths):
            edge_load[idx] += q[d]

        node_load = np.bincount(
            np.asarray(placement, dtype=int),
            weights=np.asarray(problem.needed_resources, dtype=float),
            minlength=problem.network_graph.nx_graph.number_of_nodes(),
        )
        return paths, edge_load, node_load

    def test_compact_solution(self, placement, routing, quantities) -> bool:
        try:
            x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
            validate_arc_solution(self.problem_instance, x, y, q)
        except (IndexError, ValueError, TypeError, KeyError, nx.NetworkXException):
            return False
        return True

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
        problem = self.problem_instance
        x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
        validate_arc_solution(problem, x, y, q)
        network = problem.network_graph
        edge_load = x.T @ q
        alpha = np.asarray(network.alphas)
        latency_slack = np.asarray(problem.latencies) - x @ (
            np.asarray(network.betas) + alpha * edge_load
        )
        sensitivity = x @ (alpha * x[demand_index])
        affected = sensitivity > 0
        latency_room = (
            np.min(latency_slack[affected] / sensitivity[affected])
            if np.any(affected) else np.inf
        )
        used = x[demand_index] > 0
        capacity_room = (
            np.min((np.asarray(network.capacities) - edge_load)[used])
            if np.any(used) else np.inf
        )
        flow_room = problem.max_flows[demand_index] - q[demand_index]
        return max(0.0, np.floor(min(latency_room, capacity_room, flow_room)))

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
