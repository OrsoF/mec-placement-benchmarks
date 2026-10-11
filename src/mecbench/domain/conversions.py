"""Pure compact/state/arc representation conversions."""

import networkx as nx
import numpy as np

from mecbench.domain.graph_utils import path_node_to_path_tuple


def to_int_array(x) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=float)).astype(int)


def split_state(problem, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(s)
    S = problem.number_of_service
    D = problem.total_number_of_demands
    return s[:S], s[S : S + D], s[S + D : S + 2 * D]


def make_state(placement, routing, quantities) -> np.ndarray:
    return to_int_array(np.concatenate((placement, routing, quantities)))


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
