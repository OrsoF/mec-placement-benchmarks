import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.probleminstance import ProblemInstance
from rl.path_cache import ShortestPathCache


def state_to_vectors(problem_instance: ProblemInstance, state: list) -> tuple:
    """Split [placements | path indices | quantities] into its three parts."""
    service_count = problem_instance.number_of_service
    demand_count = problem_instance.total_number_of_demands
    service_placement = state[:service_count]
    path_indices = state[service_count : service_count + demand_count]
    quantities = state[service_count + demand_count :]
    return service_placement, path_indices, quantities


def state_to_variables(
    problem_instance: ProblemInstance,
    graph,
    network_graph,
    path_cache: ShortestPathCache,
    state: list,
) -> tuple:
    """Convert environment state into arc, placement, and quantity arrays."""
    service_placement, path_indices, quantities = state_to_vectors(
        problem_instance, state
    )

    x_var = np.zeros(
        (
            problem_instance.total_number_of_demands,
            graph.number_of_edges(),
        )
    )
    for demand_index in range(problem_instance.total_number_of_demands):
        service_index = problem_instance.demand_to_service[demand_index]
        origin = problem_instance.origins[demand_index]
        target = service_placement[service_index]
        demand_path = path_node_to_path_tuple(
            path_cache.nth_shortest_path(origin, target, path_indices[demand_index])
        )
        for edge in demand_path:
            edge_index = network_graph.arc_index_dictionary[edge]
            x_var[demand_index, edge_index] = 1.0

    y_var = np.zeros((problem_instance.number_of_service, graph.number_of_nodes()))
    for service_index in range(problem_instance.number_of_service):
        y_var[service_index, service_placement[service_index]] = 1.0

    q_var = np.array(quantities)
    return x_var, y_var, q_var

