import numpy as np
import networkx as nx

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.network_graph import NetworkGraph
from rl.path_cache import ShortestPathCache


def to_integer_state(s: np.ndarray | list) -> np.ndarray:
    """Return a rounded integer copy of a flat state."""
    return np.rint(np.asarray(s, dtype=float)).astype(int)


def state_to_vectors(
    problem: ProblemInstance,
    s: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a flat state into placement, routing, and quantity vectors."""
    s = np.asarray(s)
    S = problem.number_of_service
    D = problem.total_number_of_demands
    return s[:S], s[S : S + D], s[S + D : S + 2 * D]


def make_basic_state(problem: ProblemInstance, graph: nx.Graph) -> np.ndarray:
    """Create a default state that places all services on the largest node."""
    S = problem.number_of_service
    D = problem.total_number_of_demands
    resources = np.fromiter(
        (graph.nodes[v]["available_resources"] for v in graph.nodes),
        dtype=float,
        count=graph.number_of_nodes(),
    )
    cloud = int(np.argmax(resources))
    return np.concatenate(
        (
            np.full(S, cloud, dtype=int),
            np.zeros(D, dtype=int),
            np.asarray(problem.min_flows, dtype=int),
        )
    )


def clip_state_to_bounds(
    problem: ProblemInstance,
    graph: nx.Graph,
    max_path_candidates: int,
    s: np.ndarray | list,
) -> np.ndarray:
    """Round and clip a flat state to valid state bounds."""
    s = to_integer_state(s)
    p, r, q = state_to_vectors(problem, s)
    p[:] = np.clip(p, 0, graph.number_of_nodes() - 1)
    r[:] = np.clip(r, 0, max_path_candidates - 1)
    q[:] = np.clip(q, problem.min_flows, problem.max_flows)
    return s


def make_state(
    placement: np.ndarray | list,
    routing: np.ndarray | list,
    quantities: np.ndarray | list,
) -> np.ndarray:
    """Build a flat integer state from placement, routing, and quantities."""
    return to_integer_state(np.concatenate((placement, routing, quantities)))


def is_feasible_state(
    problem: ProblemInstance,
    solution_checker: object,
    path_cache: ShortestPathCache,
    s: np.ndarray | list,
) -> bool:
    """Return whether a flat state is feasible for the problem."""
    placement, routing, quantities = state_to_vectors(problem, to_integer_state(s))
    for demand_index in range(problem.total_number_of_demands):
        service_index = problem.demand_to_service[demand_index]
        origin = problem.origins[demand_index]
        target = placement[service_index]
        try:
            path_cache.nth_shortest_path(origin, target, routing[demand_index])
        except (IndexError, nx.NetworkXException):
            return False
    return solution_checker.test_solution(placement, routing, quantities)


def state_to_variables(
    problem: ProblemInstance,
    graph: nx.Graph,
    network_graph: NetworkGraph,
    path_cache: ShortestPathCache,
    s: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a flat state into arc, placement, and quantity variables."""
    s = to_integer_state(s)
    p, r, q = state_to_vectors(problem, s)

    x = np.zeros((problem.total_number_of_demands, graph.number_of_edges()))
    for d in range(problem.total_number_of_demands):
        k = problem.demand_to_service[d]
        path = path_cache.nth_shortest_path(problem.origins[d], p[k], r[d])
        for e in path_node_to_path_tuple(path):
            x[d, network_graph.arc_index_dictionary[e]] = 1.0

    y = np.zeros((problem.number_of_service, graph.number_of_nodes()))
    y[np.arange(problem.number_of_service), p] = 1.0
    return x, y, np.asarray(q)
