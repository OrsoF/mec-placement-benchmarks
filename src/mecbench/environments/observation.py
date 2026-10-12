from typing import Any

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from mecbench.domain.loads import residual_arrays
from mecbench.domain.problem import ProblemInstance
from mecbench.environments.state_codec import state_to_vectors
from mecbench.utils.solver_paths import ShortestPathCache
from mecbench.utils.solver_workspace import apply_residuals_to_graph


def make_action_space(
    problem_instance: ProblemInstance, graph: nx.Graph, max_path_candidates: int
) -> MultiDiscrete:
    """Build the discrete action space for the loaded problem dimensions."""
    return MultiDiscrete(
        [
            3,
            problem_instance.number_of_service,
            graph.number_of_nodes(),
            problem_instance.total_number_of_demands,
            3,
            max_path_candidates,
        ]
    )


def make_vector_observation_space(
    problem_instance: ProblemInstance, graph: nx.Graph, max_path_candidates: int
) -> Box:
    """Build the vector observation bounds for the loaded problem."""
    demand_count = problem_instance.total_number_of_demands
    service_count = problem_instance.number_of_service
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    total_max_flow = sum(problem_instance.max_flows)
    max_edge_latency = max(
        (
            graph.edges[edge]["beta"] + total_max_flow * graph.edges[edge]["alpha"]
            for edge in graph.edges
        ),
        default=0.0,
    )
    latency_ratio_high = [
        node_count * max_edge_latency / max(latency, 1e-9) for latency in problem_instance.latencies
    ]

    low_array = np.zeros(
        service_count
        + demand_count
        + demand_count
        + demand_count
        + edge_count
        + node_count
        + demand_count
        + demand_count
        + edge_count
        + node_count
        + service_count
        + demand_count
    )
    high_array = np.array(
        [max(node_count - 1, 0)] * service_count
        + [max(max_path_candidates - 1, 0)] * demand_count
        + list(problem_instance.max_flows)
        + latency_ratio_high
        + [1.0] * edge_count
        + [1.0] * node_count
        + list(problem_instance.max_flows)
        + list(problem_instance.latencies)
        + list(problem_instance.network_graph.capacities)
        + list(problem_instance.network_graph.available_resources)
        + list(problem_instance.needed_resources)
        + [node_count] * demand_count
    )
    return Box(
        low=low_array.astype(np.float32),
        high=high_array.astype(np.float32),
        dtype=np.float32,
    )


def make_vector_observation(env: Any, state: np.ndarray | list) -> np.ndarray:
    """Build the flat observation vector used by current RL agents."""
    x, y, q = env._state_to_variables(state)
    residual = residual_arrays(env.problem_instance, x, y, q)
    apply_residuals_to_graph(env.problem_instance, env.graph, x, y, q)

    latency_proportion = (
        residual["demand_latency"] / np.asarray(env.problem_instance.latencies)
    ).tolist()

    arc_saturation_proportion = [
        residual["remaining_capacity"][edge_index]
        / env.problem_instance.network_graph.capacities[edge_index]
        for edge_index in range(env.graph.number_of_edges())
    ]

    server_saturation = [
        residual["remaining_resources"][node]
        / (env.problem_instance.network_graph.available_resources[node] + 0.01)
        for node in range(env.graph.number_of_nodes())
    ]

    path_lengths = [
        len(get_demand_path(env.problem_instance, env.path_cache, demand_index, state))
        for demand_index in range(env.problem_instance.total_number_of_demands)
    ]

    return np.array(
        np.asarray(state).tolist()
        + latency_proportion
        + arc_saturation_proportion
        + server_saturation
        + env.problem_instance.max_flows
        + env.problem_instance.latencies
        + env.problem_instance.network_graph.capacities
        + env.problem_instance.network_graph.available_resources
        + env.problem_instance.needed_resources
        + path_lengths,
        dtype=np.float32,
    )


def get_demand_path(
    problem: ProblemInstance,
    path_cache: ShortestPathCache,
    demand_index: int,
    state: np.ndarray | list,
) -> list:
    """Return the selected path for a demand in the given state."""
    service_placement, path_indices, _ = state_to_vectors(problem, state)
    service_index = problem.demand_to_service[demand_index]
    origin = problem.origins[demand_index]
    target = service_placement[service_index]
    return path_cache.nth_shortest_path(origin, target, path_indices[demand_index])
