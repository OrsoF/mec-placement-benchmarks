"""Parameters, optimizer options and canonical extraction for mathematical solvers.

This module does not import Pyomo, so basic usage stays independent of exact extras.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import ArcArrays, ProblemData


@dataclass(frozen=True)
class ProblemArrays:
    S: int
    D: int
    U: int
    A: int
    origins: Sequence[int]
    qmin: Sequence[float]
    qmax: Sequence[float]
    latencies: Sequence[float]
    resources: Sequence[float]
    capacities: Sequence[float]
    alpha: Sequence[float]
    beta: Sequence[float]
    demand_to_service: Sequence[int]
    arc_index: Mapping[tuple[int, int], int]
    arc_list: Sequence[tuple[int, int]]
    node_resources: Sequence[float]


def problem_arrays(problem: ProblemData) -> ProblemArrays:
    graph = problem.network_graph.nx_graph
    return ProblemArrays(
        S=problem.number_of_service,
        D=problem.total_number_of_demands,
        U=graph.number_of_nodes(),
        A=graph.number_of_edges(),
        origins=problem.origins,
        qmin=problem.min_flows,
        qmax=problem.max_flows,
        latencies=problem.latencies,
        resources=problem.needed_resources,
        capacities=problem.network_graph.capacities,
        alpha=problem.network_graph.alphas,
        beta=problem.network_graph.betas,
        demand_to_service=problem.demand_to_service,
        arc_index=problem.network_graph.arc_index_dictionary,
        arc_list=problem.network_graph.arc_list,
        node_resources=problem.network_graph.available_resources,
    )


def configure_gurobi(
    opt: Any,
    options: Mapping[str, Any] | None = None,
    time_limit: float | None = None,
    thread_limit: int | None = None,
) -> Any:
    if time_limit is not None:
        opt.options["TimeLimit"] = time_limit
    if thread_limit is not None:
        opt.options["threads"] = thread_limit
    for key, value in (options or {}).items():
        opt.options[key] = value
    return opt


def objective_value(q: ArrayLike) -> float:
    return float(np.sum(np.asarray(q, dtype=float)))


def variable_value(item: Any) -> float:
    value = getattr(item, "value", item)
    return float("nan") if value is None else float(value)


def arc_model_arrays(
    x: Any,
    y: Any,
    q: Any,
    demand_count: int,
    arc_count: int,
    service_count: int,
    node_count: int,
) -> ArcArrays:
    return (
        np.array(
            [[variable_value(x[d, a]) for a in range(arc_count)] for d in range(demand_count)],
            dtype=float,
        ).reshape(demand_count, arc_count),
        np.array(
            [[variable_value(y[s, u]) for u in range(node_count)] for s in range(service_count)],
            dtype=float,
        ).reshape(service_count, node_count),
        np.array([variable_value(q[d]) for d in range(demand_count)], dtype=float),
    )


def path_model_arrays(
    weights: Any,
    y: Any,
    q: Any,
    paths: Sequence[Sequence[int]],
    path_to_demand: Sequence[int],
    arc_index: Mapping[tuple[int, int], int],
    demand_count: int,
    arc_count: int,
    service_count: int,
    node_count: int,
) -> ArcArrays:
    x = np.zeros((demand_count, arc_count))
    for p, path in enumerate(paths):
        weight = variable_value(weights[p])
        d = path_to_demand[p]
        for source, destination in zip(path, path[1:], strict=False):
            x[d, arc_index[source, destination]] += weight
    placement = np.array(
        [[variable_value(y[s, u]) for u in range(node_count)] for s in range(service_count)],
        dtype=float,
    ).reshape(service_count, node_count)
    quantities = np.array([variable_value(q[d]) for d in range(demand_count)], dtype=float)
    return x, placement, quantities
