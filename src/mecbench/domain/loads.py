"""Pure load and residual calculations."""

import numpy as np


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
