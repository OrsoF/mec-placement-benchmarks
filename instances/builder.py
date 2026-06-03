import networkx as nx
import numpy as np

from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance

# graph builders
from instances.graph_build import (
    build_tree_graph,
    build_star_graph,
    build_large_graph,
)

# service builders
from instances.package_build import (
    build_tree_service_package,
    build_star_service_package,
    build_large_service_package,
)


# ============================================================
# Helpers
# ============================================================

def _constant_edge_list(
    graph: nx.Graph,
    value: float,
) -> list[float]:

    return [value] * graph.number_of_edges()


def _make_network_graph(
    graph: nx.Graph,
    node_resources: list[float],
    edge_capacities: list[float],
    alpha: float,
    beta: float,
) -> NetworkGraph:

    return NetworkGraph(
        base_graph=graph,
        node_resources=node_resources,
        edge_capacity=edge_capacities,
        latency_alpha_factor=_constant_edge_list(graph, alpha),
        latency_beta_factor=_constant_edge_list(graph, beta),
    )


# ============================================================
# Tree instance
# ============================================================

def create_tree_problem_instance(
    n_users: int,
    neighbors: int = 1,
) -> ProblemInstance:

    graph = build_tree_graph(
        n_users=n_users,
        neighbors=neighbors,
    )

    node_resources = (
        [10] * n_users
        + [50]      # wap
        + [1000]    # cloud
    )

    edge_capacities = [50] * graph.number_of_edges()

    network_graph = _make_network_graph(
        graph=graph,
        node_resources=node_resources,
        edge_capacities=edge_capacities,
        alpha=3.0,
        beta=3.0,
    )

    instance_name = f"tree_{n_users}_{neighbors}"

    service_package = build_tree_service_package(
        network_graph=network_graph,
        n_services=n_users,
        instance_name=instance_name,
    )

    return ProblemInstance(
        network_graph=network_graph,
        service_package=service_package,
        problem_name=instance_name,
    )


# ============================================================
# Star/grouped instance
# ============================================================

def create_star_problem_instance(
    n_groups: int,
    users_per_group: int,
    neighbors: int = 1,
) -> ProblemInstance:

    graph = build_star_graph(
        n_groups=n_groups,
        users_per_group=users_per_group,
        neighbors=neighbors,
    )

    n_users = n_groups * users_per_group
    n_waps = n_groups

    node_resources = (
        [1] * n_users
        + [users_per_group] * n_waps
        + [n_users]
    )

    edge_capacities = [users_per_group] * graph.number_of_edges()

    network_graph = _make_network_graph(
        graph=graph,
        node_resources=node_resources,
        edge_capacities=edge_capacities,
        alpha=3.0,
        beta=3.0,
    )

    instance_name = (
        f"star_{n_groups}_{users_per_group}_{neighbors}"
    )

    service_package = build_star_service_package(
        network_graph=network_graph,
        n_groups=n_groups,
        users_per_group=users_per_group,
        instance_name=instance_name,
    )

    return ProblemInstance(
        network_graph=network_graph,
        service_package=service_package,
        problem_name=instance_name,
    )


# ============================================================
# Large hierarchical instance
# ============================================================

def create_large_problem_instance(
    base_graph: nx.Graph,
    n_services: int,
    demands_per_service: int,
    edge_per_core: int = 10,
    users_per_edge: int = 2,
    rng: np.random.Generator | None = None,
) -> ProblemInstance:

    rng = rng or np.random.default_rng(0)

    graph = build_large_graph(
        base_graph=base_graph,
        edge_per_core=edge_per_core,
        users_per_edge=users_per_edge,
        rng=rng,
    )

    node_resources = []

    for node in graph.nodes:

        layer = graph.nodes[node].get("layer")

        if layer == "edge":
            node_resources.append(1e4)

        elif layer == "user":
            node_resources.append(1e2)

        else:
            node_resources.append(1e6)

    edge_capacities = []

    for u, v in graph.edges:

        u_is_core = graph.nodes[u].get("layer") is None
        v_is_core = graph.nodes[v].get("layer") is None

        if u_is_core and v_is_core:
            edge_capacities.append(5000)

        elif u_is_core or v_is_core:
            edge_capacities.append(3000)

        else:
            edge_capacities.append(500)

    network_graph = _make_network_graph(
        graph=graph,
        node_resources=node_resources,
        edge_capacities=edge_capacities,
        alpha=0.5,
        beta=5.0,
    )

    instance_name = (
        f"large_{n_services}_{demands_per_service}"
    )

    service_package = build_large_service_package(
        network_graph=network_graph,
        n_services=n_services,
        demands_per_service=demands_per_service,
        instance_name=instance_name,
        rng=rng,
    )

    return ProblemInstance(
        network_graph=network_graph,
        service_package=service_package,
        problem_name=instance_name,
    )
