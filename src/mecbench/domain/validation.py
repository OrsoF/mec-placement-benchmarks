"""Solver-independent feasibility checks for binary arc/placement solutions.

The physical constraints match the arc MILP: x and y are binary, while q
may be continuous. Flow conservation permits balanced circulations, just as
the formulation does; no additional simple-path constraint is imposed.
"""

import numpy as np


class SolutionValidationError(ValueError):
    """A solution violates its dimensions, variable domains, or constraints."""


def _array(values, shape, name):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise SolutionValidationError(f"{name} must be a numeric array.") from error
    if result.shape != shape:
        raise SolutionValidationError(f"{name} shape must be {shape}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise SolutionValidationError(f"{name} must contain only finite values.")
    return result


def _binary(values, name, atol):
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > atol) or np.any(
        (rounded < 0) | (rounded > 1)
    ):
        raise SolutionValidationError(f"{name} must be binary within tolerance {atol}.")
    return rounded


def validate_arc_solution(problem, x, y, q, *, atol=1e-6) -> None:
    """Raise SolutionValidationError on the first feasibility violation.

    Arc and node columns use the instance's arc_list and node indices.
    Binary solver noise is rounded within ``atol``; inequalities use the
    same absolute tolerance. Relaxed x/y solutions are deliberately rejected.
    Latency is x @ (beta + alpha * (x.T @ q)), including shared arc traffic.
    This function does not mutate the instance or input arrays.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    network = problem.network_graph
    graph = network.nx_graph
    demands = problem.total_number_of_demands
    services = problem.number_of_service
    nodes = graph.number_of_nodes()
    arcs = graph.number_of_edges()
    x = _binary(_array(x, (demands, arcs), "x"), "x", atol)
    y = _binary(_array(y, (services, nodes), "y"), "y", atol)
    q = _array(q, (demands,), "q")

    if np.any(y.sum(axis=1) != 1):
        raise SolutionValidationError("Each service must be placed on exactly one node.")
    if np.any(q < -atol) or np.any(q < np.asarray(problem.min_flows) - atol):
        raise SolutionValidationError("Demand quantity below minimum flow.")
    if np.any(q > np.asarray(problem.max_flows) + atol):
        raise SolutionValidationError("Demand quantity above maximum flow.")

    node_load = y.T @ np.asarray(problem.needed_resources, dtype=float)
    if np.any(node_load > np.asarray(network.available_resources) + atol):
        raise SolutionValidationError("Node resources exceeded.")
    edge_load = x.T @ q
    if np.any(edge_load > np.asarray(network.capacities) + atol):
        raise SolutionValidationError("Arc capacity exceeded.")

    # Sum outgoing minus incoming routing indicators at every node. This
    # also checks empty routes and routes when origin equals placement.
    balance = np.zeros((demands, nodes))
    for a, (source, destination) in enumerate(network.arc_list):
        balance[:, source] += x[:, a]
        balance[:, destination] -= x[:, a]
    expected = -y[np.asarray(problem.demand_to_service, dtype=int)].copy()
    expected[np.arange(demands), np.asarray(problem.origins, dtype=int)] += 1
    if np.any(np.abs(balance - expected) > atol):
        raise SolutionValidationError("Flow conservation violated.")

    arc_latency = np.asarray(network.betas) + np.asarray(network.alphas) * edge_load
    demand_latency = x @ arc_latency
    if np.any(demand_latency > np.asarray(problem.latencies) + atol):
        raise SolutionValidationError("Demand latency exceeded under total arc load.")
