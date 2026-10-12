"""Reusable initial candidates, clipping, repair and quantity search."""

import networkx as nx
import numpy as np

from mecbench.domain.conversions import make_state, split_state, to_int_array
from mecbench.domain.loads import traffic_arrays
from mecbench.domain.validation import validate_arc_solution


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


def remaining_quantity(problem, x, y, q, demand_index):
    """Integer increase allowed by every affected demand's shared-arc latency."""
    validate_arc_solution(problem, x, y, q)
    network = problem.network_graph
    traffic = traffic_arrays(problem, x, q)
    edge_load = traffic["edge_load"]
    alpha = np.asarray(network.alphas)
    latency_slack = np.asarray(problem.latencies) - traffic["demand_latency"]
    sensitivity = x @ (alpha * x[demand_index])
    affected = sensitivity > 0
    latency_room = (
        np.min(latency_slack[affected] / sensitivity[affected]) if np.any(affected) else np.inf
    )
    used = x[demand_index] > 0
    capacity_room = (
        np.min((np.asarray(network.capacities) - edge_load)[used]) if np.any(used) else np.inf
    )
    flow_room = problem.max_flows[demand_index] - q[demand_index]
    return max(0.0, np.floor(min(latency_room, capacity_room, flow_room)))
