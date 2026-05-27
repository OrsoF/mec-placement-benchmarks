import numpy as np
from gymnasium.spaces import Box, MultiDiscrete


def make_action_space(problem_instance, graph, max_pcc: int) -> MultiDiscrete:
    return MultiDiscrete(
        [
            3,
            problem_instance.number_of_service,
            graph.number_of_nodes(),
            problem_instance.total_number_of_demands,
            3,
            max_pcc,
        ]
    )


def make_vector_observation_space(problem_instance, graph, max_pcc: int) -> Box:
    high = max(
        graph.number_of_nodes(),
        max_pcc,
        problem_instance.max_flows[0],
    )
    observation_shape = (
        problem_instance.number_of_service
        + problem_instance.total_number_of_demands
        + problem_instance.total_number_of_demands
        + problem_instance.total_number_of_demands
        + graph.number_of_edges()
        + graph.number_of_nodes()
        + problem_instance.total_number_of_demands
        + problem_instance.total_number_of_demands
        + graph.number_of_edges()
        + graph.number_of_nodes()
        + problem_instance.number_of_service
        + problem_instance.total_number_of_demands
    )
    low_array = np.zeros(observation_shape)
    high_array = high * np.ones(observation_shape)
    return Box(
        low=low_array.astype(np.float32),
        high=high_array.astype(np.float32),
        dtype=np.float32,
    )


def observe_vector(env, state: list) -> np.ndarray:
    """Build the flat observation vector used by current RL agents."""
    env._reset_edge_attributes()
    env._reset_node_attributes()
    env._update_edge_attributes(state)
    env._update_node_attributes(state)

    latency_proportion = [
        env._get_latency_demand(demand_index, state)
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
        len(env._get_path_demand(demand_index, state))
        for demand_index in range(env.problem_instance.total_number_of_demands)
    ]

    return np.array(
        state
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

