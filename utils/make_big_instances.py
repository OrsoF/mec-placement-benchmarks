from networkx.readwrite.gml import read_gml, write_gml
import networkx as nx
import os
import numpy as np
import pickle
from problem_classes.networkgraph import NetworkGraph
from problem_classes.demand import Demand
from problem_classes.service import Service
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import pickle_save
from utils.graph_utils import make_graph_from_gml


def generate_one_service(
    network_graph: NetworkGraph,
    demand_number: int,
    maximum_latency: float,
    resources_needed: float,
    flow: float,
) -> Service:
    task_list = []
    for _ in range(demand_number):
        origin_node = np.random.choice(network_graph.nx_graph.nodes)
        task = Demand(
            origin_node_index=origin_node,
            flow_min=1,
            flow_max=flow,
            max_latency=maximum_latency,
            resources_needed=resources_needed,
        )
        task_list.append(task)
    return Service(task_list, network_graph)


def generate_experimental_service_package(
    network_graph: NetworkGraph, service_number: int, graph_name: str
) -> ServicePackage:
    nx_graph = network_graph.nx_graph
    average_alpha = sum(nx_graph.edges[edge]["alpha"] for edge in nx_graph.edges) / len(
        nx_graph.edges
    )
    average_beta = sum(nx_graph.edges[edge]["beta"] for edge in nx_graph.edges) / len(
        nx_graph.edges
    )
    max_latency = (
        7.0
        * (average_alpha * 5.0 * service_number**2 / len(nx_graph.edges) + average_beta)
        * 1e3
    )
    services = [
        generate_one_service(
            network_graph=network_graph,
            demand_number=service_number,
            maximum_latency=max_latency,
            resources_needed=5000 // service_number,
            flow=100.0,
        )
        for _ in range(service_number)
    ]

    instance_name = "experimental_{}_{}_{}".format(
        graph_name, service_number, service_number
    )

    return ServicePackage(services, instance_name)


def transform_base_graph(nx_graph: nx.Graph) -> NetworkGraph:
    resources = [10**4 for _ in nx_graph.nodes]
    capacities = [200 + 200 * len(nx_graph.nodes) // 15 for _ in nx_graph.edges]
    alphas = [0.2 for _ in nx_graph.edges]
    betas = [5.0 for _ in nx_graph.edges]
    return NetworkGraph(nx_graph, resources, capacities, alphas, betas)


def add_two_levels_to_base_graph(network_graph: NetworkGraph) -> NetworkGraph:
    base_nx_graph = network_graph.nx_graph
    node_number = len(base_nx_graph.nodes)

    # Create 10*n edge servers
    new_nodes_labels = list(range(node_number, 11 * node_number))
    base_nx_graph.add_nodes_from(new_nodes_labels)

    # Connect each edge server to a given server
    server_edge_server_connection = [
        (new_nodes_labels[i], np.random.randint(node_number))
        for i in range(10 * node_number)
    ]
    server_edge_server_connection += [
        (arc[1], arc[0]) for arc in server_edge_server_connection
    ]
    base_nx_graph.add_edges_from(server_edge_server_connection)

    # Create 20*n users
    second_labels = list(range(11 * node_number, 31 * node_number))
    base_nx_graph.add_nodes_from(second_labels)

    # Connect each user with four random edges servers
    new_connections = [
        (second_labels[i], np.random.randint(node_number, 11 * node_number))
        for i in range(20 * node_number)
        for _ in range(4)
    ]
    new_connections += [(arc[1], arc[0]) for arc in new_connections]
    base_nx_graph.add_edges_from(new_connections)

    resources = (
        [10**6] * node_number + [10**4] * 10 * node_number + [10**2] * 20 * node_number
    )

    capacities = [0] * len(base_nx_graph.edges)

    arc_dic = {edge: i for i, edge in enumerate(base_nx_graph.edges)}

    for edge in base_nx_graph.edges:
        node1, node2 = edge
        if node1 < node_number and node2 < node_number:
            capacities[arc_dic[(node1, node2)]] = 5000
        elif node1 < node_number or node2 < node_number:
            capacities[arc_dic[(node1, node2)]] = 3000
        else:
            capacities[arc_dic[(node1, node2)]] = 100

    beta = [0] * len(base_nx_graph.edges)

    for edge in base_nx_graph.edges:
        node1, node2 = edge
        if node1 < node_number and node2 < node_number:
            beta[arc_dic[(node1, node2)]] = 5
        elif node1 < node_number or node2 < node_number:
            beta[arc_dic[(node1, node2)]] = 7
        else:
            beta[arc_dic[(node1, node2)]] = 15

    alpha = [0] * len(base_nx_graph.edges)

    for edge in base_nx_graph.edges:
        node1, node2 = edge
        if node1 < node_number and node2 < node_number:
            alpha[arc_dic[(node1, node2)]] = 5 * 10**-1
        elif node1 < node_number or node2 < node_number:
            alpha[arc_dic[(node1, node2)]] = 7 * 10**-1
        else:
            alpha[arc_dic[(node1, node2)]] = 15 * 10**-1

    return NetworkGraph(base_nx_graph, resources, capacities, alpha, beta)


def generate_one_service_for_three_levels_graph(
    network_graph: NetworkGraph,
    task_number: int,
    latency: float,
    resources: float,
    flow: float,
):
    nx_graph = network_graph.nx_graph
    node_number = len(nx_graph.nodes) // 31
    possible_origins = list(range(node_number)) + list(
        range(21 * node_number, 31 * node_number)
    )
    task_list = []
    for _ in range(task_number):
        origin_node = np.random.choice(possible_origins)
        task = Demand(
            origin_node_index=origin_node,
            flow_min=1.0,
            flow_max=flow,
            max_latency=latency,
        )
        task_list.append(task)
    return Service(
        task_list=task_list, network_graph=network_graph, resources_needed=resources
    )


def generate_service_package_for_three_levels_graph(
    network_graph: NetworkGraph, service_number: int, graph_name: str
) -> ServicePackage:
    nx_graph = network_graph.nx_graph
    average_alpha = sum(nx_graph.edges[edge]["alpha"] for edge in nx_graph.edges) / len(
        nx_graph.edges
    )
    average_beta = sum(nx_graph.edges[edge]["beta"] for edge in nx_graph.edges) / len(
        nx_graph.edges
    )

    services = []
    for _ in range(service_number):
        service = generate_one_service_for_three_levels_graph(
            network_graph=network_graph,
            task_number=service_number,
            latency=5.0
            * (
                average_alpha * 5.0 * service_number**2 / len(nx_graph.edges)
                + average_beta
            ),
            resources=5000 // service_number,
            flow=100.0,
        )
        services.append(service)

    instance_name = "three_levels_experimental_{}_{}_{}".format(
        graph_name, service_number, service_number
    )

    return ServicePackage(services=services, instance_name=instance_name)


def make_big_instances():
    gml_path = "gml_graphs"
    instance_path = "instances"
    files = os.listdir(gml_path)

    for gml_graph_name in files:
        gml_path = "gml_graphs"

        gml_path = os.path.join(gml_path, gml_graph_name)
        graph_name = gml_graph_name[:-4]  # france.gml -> france

        network_graph = make_graph_from_gml(gml_path)

        three_levels_network_graph = add_two_levels_to_base_graph(network_graph)

        for number_of_service in range(1, 11):
            service_package = generate_service_package_for_three_levels_graph(
                three_levels_network_graph, number_of_service, graph_name
            )

            instance_name = "{}_{}_instance_{}_{}_{}".format(
                len(three_levels_network_graph.nx_graph.edges),
                len(three_levels_network_graph.nx_graph.edges),
                number_of_service,
                number_of_service,
                graph_name,
            )

            problem_instance = ProblemInstance(
                three_levels_network_graph, service_package, instance_name
            )

            saved_file_path = os.path.join(instance_path, instance_name + ".pkl")

            pickle_save(problem_instance, saved_file_path)
