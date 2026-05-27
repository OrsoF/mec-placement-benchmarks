import networkx as nx

from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage
from core.problem_classes.solving_instance import SolvingInstance


def _validate_tree_params(number_of_users: int, neighbors_connection: int) -> None:
    assert number_of_users >= 1
    assert neighbors_connection >= 0


def _user_node(user_index: int) -> str:
    return f"user_{user_index}"


def _make_tree_topology(
    number_of_users: int,
    neighbors_connection: int,
    wap_node: str,
    cloud_node: str,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(_user_node(i) for i in range(number_of_users))

    for user_index in range(number_of_users):
        for neighbor in range(-neighbors_connection, neighbors_connection + 1):
            connected = max(0, min(number_of_users - 1, user_index + neighbor))
            if connected != user_index:
                g.add_edge(_user_node(user_index), _user_node(connected))

    g.add_nodes_from([wap_node, cloud_node])
    g.add_edges_from((wap_node, _user_node(i)) for i in range(number_of_users))
    g.add_edge(wap_node, cloud_node)
    return g


def _tree_resources(number_of_users: int) -> list[float]:
    return [10] * number_of_users + [50, 1000]


def _tree_edge_params(
    g: nx.Graph,
    number_of_users: int,
    wap_node: str,
    cloud_node: str,
) -> tuple[list[float], list[float], list[float]]:
    capacities = []
    for u, v in nx.DiGraph(g).edges:
        edge = {u, v}
        if edge == {wap_node, cloud_node}:
            capacities.append(3 * number_of_users)
        elif wap_node in edge:
            capacities.append(50)
        else:
            capacities.append(10 * number_of_users)

    return capacities, [3.0] * len(capacities), [3.0] * len(capacities)


def _make_tree_network(
    number_of_users: int,
    neighbors_connection: int,
    wap_node: str,
    cloud_node: str,
) -> NetworkGraph:
    g = _make_tree_topology(
        number_of_users,
        neighbors_connection,
        wap_node,
        cloud_node,
    )
    capacities, latency_alphas, latency_betas = _tree_edge_params(
        g,
        number_of_users,
        wap_node,
        cloud_node,
    )
    return NetworkGraph(
        base_graph=g,
        node_resources=_tree_resources(number_of_users),
        edge_capacity=capacities,
        latency_alpha_factor=latency_alphas,
        latency_beta_factor=latency_betas,
    )


def _make_user_services(
    network_graph: NetworkGraph,
    number_of_users: int,
    latency_scale: float,
) -> list[Service]:
    services = []
    cloud_index = network_graph.nx_graph.number_of_nodes() - 1
    for user_index in range(number_of_users):
        tasks = [
            Demand(
                origin_node_index=user_index,
                flow_min=1,
                flow_max=5,
                max_latency=latency_scale * number_of_users,
            ),
            Demand(
                origin_node_index=cloud_index,
                flow_min=1,
                flow_max=2,
                max_latency=latency_scale * number_of_users,
            ),
        ]
        services.append(Service(tasks, network_graph, resources_needed=1))
    return services


def make_tree_instance(
    number_of_users: int = 7,
    neighbors_connection: int = 2,
    randomize: bool = False,
) -> ProblemInstance:
    _validate_tree_params(number_of_users, neighbors_connection)
    wap_node = "WAP"
    cloud_node = "CLOUD"
    network_graph = _make_tree_network(
        number_of_users,
        neighbors_connection,
        wap_node,
        cloud_node,
    )

    problem_name = f"tree_{number_of_users}_{neighbors_connection}"
    services = _make_user_services(network_graph, number_of_users, latency_scale=10)
    service_package = ServicePackage(services, problem_name)
    problem_instance = ProblemInstance(network_graph, service_package, problem_name)

    SolvingInstance(problem_instance, problem_name).test_basic_placement()
    return problem_instance
