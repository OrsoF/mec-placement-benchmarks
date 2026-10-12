"""Shared initial candidates, clipping, repair and quantity search."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mecbench.domain.contracts import FloatArray, ProblemData
from mecbench.domain.conversions import make_state, split_state, to_int_array
from mecbench.domain.loads import traffic_arrays
from mecbench.domain.solution import Solution
from mecbench.domain.validation import validate_arc_solution

CompactSolution: TypeAlias = tuple[list[int], list[int], list[float]]
StateArray: TypeAlias = NDArray[np.int_]


class SolutionChecker(Protocol):
    def test_solution(
        self, placement: ArrayLike, routing: ArrayLike, quantities: ArrayLike
    ) -> bool: ...


def baseline_solution(problem: ProblemData) -> Solution:
    network = problem.network_graph
    graph = network.nx_graph
    placement_node = int(np.argmax(network.available_resources))
    x = np.zeros((problem.total_number_of_demands, graph.number_of_edges()))
    for d, origin in enumerate(problem.origins):
        nodes = nx.shortest_path(graph, origin, placement_node)
        for edge in zip(nodes, nodes[1:], strict=False):
            x[d, network.arc_index_dictionary[edge]] = 1.0
    y = np.zeros((problem.number_of_service, graph.number_of_nodes()))
    y[:, placement_node] = 1.0
    return Solution(x, y, np.asarray(problem.min_flows, dtype=float))


def basic_solution(problem: ProblemData, graph: nx.Graph) -> CompactSolution:
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


def basic_state(problem: ProblemData, graph: nx.Graph) -> StateArray:
    return make_state(*basic_solution(problem, graph))


def clip_solution(
    problem: ProblemData,
    graph: nx.Graph,
    max_path: int,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> CompactSolution:
    n = graph.number_of_nodes()
    p: list[int] = np.clip(placement, 0, n - 1).astype(int).tolist()
    r: list[int] = np.clip(routing, 0, max_path - 1).astype(int).tolist()
    q: list[float] = np.clip(quantities, problem.min_flows, problem.max_flows).astype(int).tolist()
    return p, r, q


def clip_state(problem: ProblemData, graph: nx.Graph, max_path: int, s: ArrayLike) -> StateArray:
    s = to_int_array(s)
    p, r, q = split_state(problem, s)
    p[:], r[:], q[:] = clip_solution(problem, graph, max_path, p, r, q)
    return s


def greedy_quantity_lift(
    problem: ProblemData,
    checker: SolutionChecker,
    rng: np.random.Generator,
    placement: Sequence[int],
    routing: Sequence[int],
    quantities: Sequence[float],
    steps: int,
    increment: float,
) -> tuple[Sequence[int], Sequence[int], list[float]]:
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


def repair_solution(
    problem: ProblemData,
    graph: nx.Graph,
    checker: SolutionChecker,
    max_path: int,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> CompactSolution | None:
    p, r, q = clip_solution(problem, graph, max_path, placement, routing, quantities)
    if checker.test_solution(p, r, q):
        return p, r, q
    q = list(problem.min_flows)
    if checker.test_solution(p, r, q):
        return p, r, q
    return None


def remaining_quantity(
    problem: ProblemData,
    x: FloatArray,
    y: FloatArray,
    q: FloatArray,
    demand_index: int,
) -> float:
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
    return float(max(0.0, np.floor(min(latency_room, capacity_room, flow_room))))
