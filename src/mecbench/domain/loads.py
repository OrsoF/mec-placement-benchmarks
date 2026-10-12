"""Shared traffic, latency, resource and residual calculations."""

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray, ProblemData


def traffic_arrays(problem: ProblemData, x: ArrayLike, q: ArrayLike) -> dict[str, FloatArray]:
    """Calculate each arc's total traffic and the resulting demand latencies."""
    routing = np.asarray(x, dtype=float)
    quantities = np.asarray(q, dtype=float)
    network = problem.network_graph
    edge_load = routing.T @ quantities
    arc_latency = (
        np.asarray(network.betas, dtype=float) + np.asarray(network.alphas, dtype=float) * edge_load
    )
    return {
        "edge_load": edge_load,
        "arc_latency": arc_latency,
        "demand_latency": routing @ arc_latency,
    }


def residual_arrays(
    problem: ProblemData, x: ArrayLike, y: ArrayLike, q: ArrayLike
) -> dict[str, FloatArray]:
    traffic = traffic_arrays(problem, x, q)
    y = np.asarray(y, dtype=float)
    node_load = y.T @ np.asarray(problem.needed_resources, dtype=float)
    capacities = np.asarray(problem.network_graph.capacities, dtype=float)
    resources = np.asarray(problem.network_graph.available_resources, dtype=float)
    return {
        **traffic,
        "node_load": node_load,
        "remaining_capacity": capacities - traffic["edge_load"],
        "remaining_resources": resources - node_load,
    }
