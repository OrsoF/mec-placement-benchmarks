from typing import Any

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from mecbench.domain.graph_utils import path_node_to_path_tuple
from mecbench.domain.loads import residual_arrays
from mecbench.domain.problem import ProblemInstance
from mecbench.environments.state_codec import state_to_vectors
from mecbench.solvers.graph_state import apply_residuals_to_graph
from mecbench.solvers.paths import ShortestPathCache


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

    latency_proportion = [
        get_demand_latency(
            env.problem_instance,
            env.graph,
            env.path_cache,
            demand_index,
            state,
        )
        / env.problem_instance.latencies[demand_index]
        for demand_index in range(env.problem_instance.total_number_of_demands)
    ]

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


def observe_vector(env: Any, state: np.ndarray | list) -> np.ndarray:
    """Backward-compatible alias for make_vector_observation."""
    return make_vector_observation(env, state)


def apply_state_to_graph(
    problem: ProblemInstance,
    graph: nx.Graph,
    path_cache: ShortestPathCache,
    state: np.ndarray | list,
) -> None:
    """Reset graph attributes, then apply state capacity and resource usage."""
    from mecbench.domain.conversions import compact_to_arc

    p, r, q = state_to_vectors(problem, state)
    x, y, q = compact_to_arc(problem, graph, path_cache.nth_shortest_path, p, r, q)
    apply_residuals_to_graph(problem, graph, x, y, q)


def reset_graph_attributes(graph: nx.Graph) -> None:
    """Restore mutable edge and node attributes."""
    for edge in graph.edges:
        graph.edges[edge]["remaining_capacity"] = graph.edges[edge]["capacity"]
        graph.edges[edge]["latency"] = graph.edges[edge]["beta"]

    for node in graph.nodes:
        graph.nodes[node]["remaining_resources"] = graph.nodes[node]["available_resources"]


def update_edge_attributes(
    problem: ProblemInstance,
    graph: nx.Graph,
    path_cache: ShortestPathCache,
    state: np.ndarray | list,
) -> None:
    """Update edge latency and remaining capacity for a state."""
    service_placement, path_indices, quantities = state_to_vectors(problem, state)

    for demand_index in range(problem.total_number_of_demands):
        service_index = problem.demand_to_service[demand_index]
        origin = problem.origins[demand_index]
        target = service_placement[service_index]
        path = path_cache.nth_shortest_path(origin, target, path_indices[demand_index])
        demand_path = path_node_to_path_tuple(path)

        for edge in demand_path:
            graph.edges[edge]["remaining_capacity"] -= quantities[demand_index]
            graph.edges[edge]["latency"] += quantities[demand_index] * graph.edges[edge]["alpha"]


def update_node_attributes(
    problem: ProblemInstance,
    graph: nx.Graph,
    state: np.ndarray | list,
) -> None:
    """Update node resources consumed by the current service placement."""
    service_placement = state_to_vectors(problem, state)[0]

    for service_index in range(problem.number_of_service):
        target_node = service_placement[service_index]
        required_resources = problem.needed_resources[service_index]
        graph.nodes[target_node]["remaining_resources"] -= required_resources


def get_demand_latency(
    problem: ProblemInstance,
    graph: nx.Graph,
    path_cache: ShortestPathCache,
    demand_index: int,
    state: np.ndarray | list,
) -> float:
    """Return the latency of one demand path in the updated graph."""
    path = get_demand_path(problem, path_cache, demand_index, state)
    demand_path = path_node_to_path_tuple(path)
    return sum(graph.edges[edge]["latency"] for edge in demand_path)


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
