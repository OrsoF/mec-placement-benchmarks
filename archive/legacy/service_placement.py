"""Allocation helper retained for the archived heuristic ILP experiment."""

import numpy as np


def greedy_service_placement(
    available_resources_in_servers: list[float],
    resources_needed_by_services: list[float],
    server_priority: list | None = None,
):
    """
    Associated a list of services with needed_resources to a list of server with available resources following a given priority.
    """
    if server_priority is None:
        server_priority = [1 for _ in range(len(available_resources_in_servers))]
    assert len(server_priority) == len(available_resources_in_servers), (
        "The priority factors is not shaped as list[server_number]."
    )
    assert np.all([priority_value > 0 for priority_value in server_priority])

    # Initialize a list to store the allocations
    service_server_association = []

    # Create a list of server indices sorted by priority in descending order
    server_indices_sorted_by_priority = sorted(
        range(len(server_priority)),
        key=lambda server_index: available_resources_in_servers[server_index]
        / server_priority[server_index],
        reverse=True,
    )

    # Iterate through each service
    for service_resources in resources_needed_by_services:
        best_server = None

        # Iterate through the sorted servers by priority
        for server_index in server_indices_sorted_by_priority:
            server_resources = available_resources_in_servers[server_index]

            # Check if the server has enough available resources for the service
            if server_resources >= service_resources:
                if best_server is None:
                    best_server = server_index

        if best_server is not None:
            # Allocate the service to the best server
            service_server_association.append(best_server)
            # Update available resources of the chosen server
            available_resources_in_servers[best_server] -= service_resources
        else:
            # No suitable server found for the service
            service_server_association.append(None)

    assert None not in service_server_association, "One service has no server."
    assert len(service_server_association) == len(resources_needed_by_services), (
        "Allocations should be shaped as the number of service to place."
    )

    return service_server_association
