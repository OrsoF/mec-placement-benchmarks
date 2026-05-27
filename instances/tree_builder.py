from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.service_package import ServicePackage
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.solving_instance import SolvingInstance
from core.problem_classes.service import Service
from core.problem_classes.demand import Demand

import networkx as nx
import numpy as np
import os

rng_gen = np.random.default_rng(seed=0)


def make_tree_instance(
    number_of_users: int = 7, neighbors_connection: int = 2, randomize: bool = False
) -> ProblemInstance:
    assert number_of_users >= 1
    assert neighbors_connection >= 0
    nx_graph = nx.Graph()
    USER_NODE_NAME = "user_{}"
    WAP_NODE = "WAP"
    CLOUD_NODE = "CLOUD"

    nx_graph.add_nodes_from(
        [USER_NODE_NAME.format(user_index) for user_index in range(number_of_users)]
    )
    for user_index in range(number_of_users):
        for neighbor in range(-neighbors_connection, neighbors_connection + 1):
            edge = (
                USER_NODE_NAME.format(user_index),
                USER_NODE_NAME.format(
                    max(0, min(number_of_users - 1, user_index + neighbor))
                ),
            )
            if edge[0] != edge[1]:
                nx_graph.add_edge(*edge)

    nx_graph.add_nodes_from([WAP_NODE, CLOUD_NODE])

    for user_index in range(number_of_users):
        nx_graph.add_edge(WAP_NODE, USER_NODE_NAME.format(user_index))

    nx_graph.add_edge(WAP_NODE, CLOUD_NODE)

    resources = [10] * number_of_users + [
        50,
        1000,
    ]

    # Capacités des arcs
    capacities = list()
    capacities += (
        [number_of_users * 10] * 2 * (nx_graph.number_of_edges() - number_of_users - 1)
    )  # Between Users
    capacities += [50] * 2 * number_of_users  # Edge - Users
    capacities += [3 * number_of_users] * 2  # Edge - Cloud

    # Build graph
    latency_alphas = [3.0] * nx_graph.number_of_edges() * 2
    latency_betas = [3.0] * nx_graph.number_of_edges() * 2
    network_graph = NetworkGraph(
        base_graph=nx_graph,
        node_resources=resources,
        edge_capacity=capacities,
        latency_alpha_factor=latency_alphas,
        latency_beta_factor=latency_betas,
    )

    # Create services, one per user
    services = []
    for user_index in range(number_of_users):
        demand_user = Demand(
            origin_node_index=user_index,
            flow_min=1,
            flow_max=5,
            max_latency=10 * number_of_users,
        )
        demand_server = Demand(
            origin_node_index=nx_graph.number_of_nodes() - 1,
            flow_min=1,
            flow_max=2,
            max_latency=10 * number_of_users,
        )
        service = Service(
            task_list=[
                demand_user,
                demand_server,
            ],
            network_graph=network_graph,
            resources_needed=1,
        )
        services.append(service)

    PROBLEM_NAME = "tree_{}_{}".format(number_of_users, neighbors_connection)
    INSTANCE_PATH = os.path.join(
        os.getcwd(),
        "artifacts",
        "instances",
        PROBLEM_NAME + ".pkl",  # + f"{np.random.randint(100)}.pkl"
    )

    service_package = ServicePackage(services, PROBLEM_NAME)
    problem_instance = ProblemInstance(network_graph, service_package, PROBLEM_NAME)

    # pickle_save(problem_instance, INSTANCE_PATH)

    solving_instance = SolvingInstance(problem_instance, PROBLEM_NAME)
    solving_instance.test_basic_placement()

    return problem_instance
