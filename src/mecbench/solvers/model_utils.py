from dataclasses import dataclass

from mecbench.domain.problem import ProblemInstance

import numpy as np


@dataclass(frozen=True)
class ProblemArrays:
    S: int
    D: int
    U: int
    A: int
    origins: list
    qmin: list
    qmax: list
    latencies: list
    resources: list
    capacities: list
    alpha: list
    beta: list
    demand_to_service: list
    arc_index: dict
    arc_list: list
    node_resources: list


def problem_arrays(problem: ProblemInstance) -> ProblemArrays:
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


def configure_gurobi(opt, options=None, time_limit=None, thread_limit=None):
    if time_limit is not None:
        opt.options["TimeLimit"] = time_limit
    if thread_limit is not None:
        opt.options["threads"] = thread_limit
    for key, value in (options or {}).items():
        opt.options[key] = value
    return opt


def objective_value(q) -> float:
    return float(np.sum(np.asarray(q, dtype=float)))
