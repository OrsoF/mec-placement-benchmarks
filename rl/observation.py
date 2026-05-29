import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
import networkx as nx
from typing import Any

from core.problem_classes.problem_instance import ProblemInstance


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
        node_count * max_edge_latency / max(latency, 1e-9)
        for latency in problem_instance.latencies
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


def make_vector_observation(env: Any, state: list) -> np.ndarray:
    """Build the flat observation vector used by current RL agents."""
    env._reset_edge_attributes()
    env._reset_node_attributes()
    env._update_edge_attributes(state)
    env._update_node_attributes(state)

    latency_proportion = [
        env._get_demand_latency(demand_index, state)
        / env.problem_instance.latencies[demand_index]
        for demand_index in range(env.problem_instance.total_number_of_demands)
    ]

    arc_saturation_proportion = [
        env.graph.edges[edge]["remaining_capacity"] / env.graph.edges[edge]["capacity"]
        for edge in env.graph.edges
    ]

    server_saturation = [
        env.graph.nodes[node]["remaining_resources"]
        / (env.graph.nodes[node]["available_resources"] + 0.01)
        for node in env.graph.nodes
    ]

    path_lengths = [
        len(env._get_demand_path(demand_index, state))
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


def observe_vector(env: Any, state: list) -> np.ndarray:
    """Backward-compatible alias for make_vector_observation."""
    return make_vector_observation(env, state)
