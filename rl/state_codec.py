from dataclasses import dataclass

import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.problem_instance import ProblemInstance
from rl.path_cache import ShortestPathCache


@dataclass(frozen=True)
class SPPState:
    """Named environment state for placement, routing, and demand quantities."""

    placements: list
    path_indices: list
    quantities: list

    def to_list(self) -> list:
        return list(self.placements) + list(self.path_indices) + list(self.quantities)

    def copy(self) -> list:
        return self.to_list()

    def __iter__(self):
        return iter(self.to_list())

    def __len__(self):
        return len(self.placements) + len(self.path_indices) + len(self.quantities)

    def __getitem__(self, item):
        return self.to_list()[item]

    @classmethod
    def from_flat(cls, problem_instance: ProblemInstance, state: list) -> "SPPState":
        placements, path_indices, quantities = state_to_vectors(problem_instance, state)
        return cls(list(placements), list(path_indices), list(quantities))


def state_to_list(state) -> list:
    if isinstance(state, SPPState):
        return state.to_list()
    return list(state)


def state_to_vectors(problem_instance: ProblemInstance, state) -> tuple:
    """Split [placements | path indices | quantities] into its three parts."""
    if isinstance(state, SPPState):
        return state.placements, state.path_indices, state.quantities

    flat_state = state_to_list(state)
    service_count = problem_instance.number_of_service
    demand_count = problem_instance.total_number_of_demands
    service_placement = flat_state[:service_count]
    path_indices = flat_state[service_count : service_count + demand_count]
    quantities = flat_state[service_count + demand_count :]
    return service_placement, path_indices, quantities


def make_basic_state(problem_instance: ProblemInstance, graph) -> SPPState:
    """Build the simple baseline state used before optional warm starts."""
    node_with_max_resources = np.argmax(
        [graph.nodes[node]["available_resources"] for node in graph.nodes]
    )
    service_placement = [
        node_with_max_resources
    ] * problem_instance.number_of_service
    path_indices = [0] * problem_instance.total_number_of_demands
    quantities = list(problem_instance.min_flows)
    return SPPState(service_placement, path_indices, quantities)


def clip_state_to_bounds(
    problem_instance: ProblemInstance,
    graph,
    max_path_candidates: int,
    state,
) -> SPPState:
    """Clip a flat or named state to action-compatible integer bounds."""
    placement, routing, quantities = state_to_vectors(problem_instance, state)
    node_count = graph.number_of_nodes()
    return SPPState(
        [
            int(np.clip(round(node), 0, node_count - 1))
            for node in placement
        ],
        [
            int(np.clip(round(path_index), 0, max_path_candidates - 1))
            for path_index in routing
        ],
        [
            int(np.clip(round(quantity), min_flow, max_flow))
            for quantity, min_flow, max_flow in zip(
                quantities,
                problem_instance.min_flows,
                problem_instance.max_flows,
            )
        ],
    )


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
