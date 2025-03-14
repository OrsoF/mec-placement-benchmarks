from networkx.readwrite.gml import read_gml
import networkx as nx
from problem_classes.networkgraph import NetworkGraph
from problem_classes.demand import Demand
from problem_classes.service import Service
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
import numpy as np
import os
from utils.pickle import pickle_save
from utils.graph_utils import make_graph_from_gml
import matplotlib.pyplot as plt

rng_gen = np.random.default_rng(seed=0)


def make_parallel_instance(
    number_of_users_per_group: int,
    neighbors_connection: int,
    number_of_groups: int,
) -> ProblemInstance:
    assert number_of_users_per_group >= 1
    assert neighbors_connection >= 0
    assert number_of_groups >= 1

    assert neighbors_connection <= number_of_users_per_group // 2

    nx_graph = nx.Graph()

    # # Nodes (resources)
    # Add users
    for group_index in range(number_of_groups):
        for user_index in range(number_of_users_per_group):
            nx_graph.add_node(f"user_{group_index}_{user_index}")
    # Add edges
    nx_graph.add_nodes_from(
        [f"wap_{group_index}" for group_index in range(number_of_groups)]
    )
    # Add cloud
    nx_graph.add_node("cloud")

    # Resources
    resources_nodes = (
        [int(rng_gen.random() < 0.5)] * number_of_groups * number_of_users_per_group
    )
    resources_wap = [rng_gen.integers(number_of_users_per_group)] * number_of_groups
    resources_cloud = [1 * number_of_groups * number_of_users_per_group]
    resources = resources_nodes + resources_wap + resources_cloud

    # # Edges
    # Between users
    for group_index in range(number_of_groups):
        for user_index in range(number_of_users_per_group):
            for connection in range(-neighbors_connection, neighbors_connection + 1):
                origin_node = f"user_{group_index}_{user_index}"
                connected_node = (user_index + connection) % number_of_users_per_group
                destination_node = f"user_{group_index}_{connected_node}"
                if origin_node != destination_node:
                    nx_graph.add_edge(origin_node, destination_node)

    # Between user and wap
    for group_index in range(number_of_groups):
        for user_index in range(number_of_users_per_group):
            nx_graph.add_edge(f"user_{group_index}_{user_index}", f"wap_{group_index}")

    # Between wap and cloud
    for group_index in range(number_of_groups):
        nx_graph.add_edge(f"wap_{group_index}", "cloud")

    # Capacities
    capacity_user_user = (
        [1 * number_of_users_per_group]
        * number_of_users_per_group
        * neighbors_connection
        * number_of_groups
    )
    capacity_user_wap = (
        [1 * number_of_users_per_group] * number_of_users_per_group * number_of_groups
    )
    capacity_wap_cloud = [
        1 * number_of_users_per_group * number_of_groups
    ] * number_of_groups
    capacities = capacity_user_user + capacity_user_wap + capacity_wap_cloud
    latency_alphas = [3.0] * nx_graph.number_of_edges()
    latency_betas = [3.0] * nx_graph.number_of_edges()

    network_graph = NetworkGraph(
        base_graph=nx_graph,
        node_resources=resources,
        edge_capacity=capacities,
        latency_alpha_factor=latency_alphas,
        latency_beta_factor=latency_betas,
    )

    # Create services, one per user
    services = []
    for user_index in range(number_of_users_per_group * number_of_groups):
        demand_user = Demand(
            origin_node_index=user_index,
            flow_min=1,
            flow_max=5,
            max_latency=5 * number_of_users_per_group * number_of_groups,
        )
        demand_server = Demand(
            origin_node_index=nx_graph.number_of_nodes() - 1,
            flow_min=1,
            flow_max=2,
            max_latency=5 * number_of_users_per_group * number_of_groups,
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

    PROBLEM_NAME = "tree_parallel_{}_{}".format(
        number_of_users_per_group * number_of_groups, neighbors_connection
    )
    INSTANCE_PATH = os.path.join(
        os.getcwd(),
        "instances",
        PROBLEM_NAME + ".pkl",  # + f"{np.random.randint(100)}.pkl"
    )
    print("Instance creation...")

    service_package = ServicePackage(services, PROBLEM_NAME)
    problem_instance = ProblemInstance(network_graph, service_package, PROBLEM_NAME)

    # pickle_save(problem_instance, INSTANCE_PATH)

    return problem_instance
