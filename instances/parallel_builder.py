import networkx as nx
import numpy as np

from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage

rng_gen = np.random.default_rng(seed=0)


def _validate_star_params(
    number_of_groups: int,
    users_per_group: int,
    neighbors_connection: int,
) -> None:
    assert number_of_groups >= 1
    assert users_per_group >= 1
    assert neighbors_connection >= 0
    assert neighbors_connection <= users_per_group // 2, "Too many neighbors."


def _user_node(group_index: int, user_index: int) -> str:
    return f"user_{group_index}_{user_index}"


def _wap_node(group_index: int) -> str:
    return f"wap_{group_index}"


def _make_star_topology(
    number_of_groups: int,
    users_per_group: int,
    neighbors_connection: int,
    cloud_node: str,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(
        _user_node(group_index, user_index)
        for group_index in range(number_of_groups)
        for user_index in range(users_per_group)
    )
    g.add_nodes_from(_wap_node(group_index) for group_index in range(number_of_groups))
    g.add_node(cloud_node)

    for group_index in range(number_of_groups):
        for user_index in range(users_per_group):
            u = _user_node(group_index, user_index)
            for offset in range(-neighbors_connection, neighbors_connection + 1):
                connected = (user_index + offset) % users_per_group
                v = _user_node(group_index, connected)
                if u != v:
                    g.add_edge(u, v)
            g.add_edge(u, _wap_node(group_index))

    g.add_edges_from(
        (_wap_node(group_index), cloud_node) for group_index in range(number_of_groups)
    )
    return g


def _star_resources(number_of_groups: int, users_per_group: int) -> list[float]:
    user_resources = [int(rng_gen.random() < 0.5)] * number_of_groups * users_per_group
    wap_resources = [rng_gen.integers(users_per_group)] * number_of_groups
    cloud_resources = [number_of_groups * users_per_group]
    return user_resources + wap_resources + cloud_resources


def _star_edge_params(
    g: nx.Graph,
    users_per_group: int,
    number_of_groups: int,
    cloud_node: str,
) -> tuple[list[float], list[float], list[float]]:
    capacities = []
    for u, v in nx.DiGraph(g).edges:
        edge = {u, v}
        if cloud_node in edge:
            capacities.append(users_per_group * number_of_groups)
        elif any(node.startswith("wap_") for node in edge):
            capacities.append(users_per_group)
        else:
            capacities.append(users_per_group)

    return capacities, [3.0] * len(capacities), [3.0] * len(capacities)


def _make_star_network(
    number_of_groups: int,
    users_per_group: int,
    neighbors_connection: int,
    cloud_node: str,
) -> NetworkGraph:
    g = _make_star_topology(
        number_of_groups,
        users_per_group,
        neighbors_connection,
        cloud_node,
    )
    capacities, latency_alphas, latency_betas = _star_edge_params(
        g,
        users_per_group,
        number_of_groups,
        cloud_node,
    )
    return NetworkGraph(
        base_graph=g,
        node_resources=_star_resources(number_of_groups, users_per_group),
        edge_capacity=capacities,
        latency_alpha_factor=latency_alphas,
        latency_beta_factor=latency_betas,
    )


def _make_star_services(
    network_graph: NetworkGraph,
    number_of_groups: int,
    users_per_group: int,
) -> list[Service]:
    n_users = number_of_groups * users_per_group
    cloud_index = network_graph.nx_graph.number_of_nodes() - 1
    services = []
    for user_index in range(n_users):
        tasks = [
            Demand(
                origin_node_index=user_index,
                flow_min=1,
                flow_max=5,
                max_latency=10 * n_users,
            ),
            Demand(
                origin_node_index=cloud_index,
                flow_min=1,
                flow_max=2,
                max_latency=10 * n_users,
            ),
        ]
        services.append(Service(tasks, network_graph, resources_needed=1))
    return services


def build_star_instance(
    number_of_groups: int,
    users_per_group: int,
    neighbors_connection: int,
) -> ProblemInstance:
    _validate_star_params(number_of_groups, users_per_group, neighbors_connection)
    cloud_node = "cloud"
    network_graph = _make_star_network(
        number_of_groups,
        users_per_group,
        neighbors_connection,
        cloud_node,
    )
    n_users = users_per_group * number_of_groups
    problem_name = f"tree_parallel_{n_users}_{neighbors_connection}"
    services = _make_star_services(network_graph, number_of_groups, users_per_group)
    service_package = ServicePackage(services, problem_name)
    return ProblemInstance(network_graph, service_package, problem_name)
