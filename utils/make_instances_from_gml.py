from networkx.readwrite.gml import read_gml
import networkx as nx
from problem_classes.networkgraph import NetworkGraph
from problem_classes.demand import Demand
from problem_classes.service import Service
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
import numpy as np
import os
from utils.data_management import pickle_save
from utils.graph_utils import make_graph_from_gml


def make_services(
    network_graph: NetworkGraph,
    number_of_demands_per_service: int,
    number_of_services: int,
    instance_name: str,
) -> ServicePackage:
    """
    Build a ServicePackage object with multiples
    services and multiple demands per services.
    """
    alpha_moyen = sum(
        network_graph.nx_graph.edges[edge]["alpha"]
        for edge in network_graph.nx_graph.edges
    ) / len(network_graph.nx_graph.edges)
    beta_moyen = sum(
        network_graph.nx_graph.edges[edge]["beta"]
        for edge in network_graph.nx_graph.edges
    ) / len(network_graph.nx_graph.edges)

    demands = [
        Demand(
            origin_node_index=np.random.choice(network_graph.nx_graph.nodes),
            flow_min=1,
            flow_max=100,
            max_latency=7.0
            * (
                alpha_moyen
                * 5.0
                * number_of_services**2
                / len(network_graph.nx_graph.edges)
                + beta_moyen
            ),
            resources_needed=5000 // number_of_services,
        )
        for _ in range(number_of_demands_per_service)
    ]

    services = [Service(demands, network_graph) for _ in range(number_of_services)]

    service_package = ServicePackage(services, instance_name)

    return service_package


all_gml_path = os.path.join(os.getcwd(), "gml_graphs")
instance_path = os.path.join(os.getcwd(), "instances")


def make_instance_from_gml(
    all_gml_path: str = all_gml_path, instance_path: str = instance_path
):
    gml_list_original = os.listdir(all_gml_path)

    for gml_graph_name in gml_list_original:
        gml_path = os.path.join(all_gml_path, gml_graph_name)
        graph_name = gml_graph_name[:-4]  # france.gml -> france

        # Transforming gml into NetworkGraph
        network_graph: NetworkGraph = make_graph_from_gml(gml_path)

        # Building services
        for number_of_service in range(1, 11):
            number_of_demands_per_service = number_of_service

            instance_name = "{}_{}_instance_{}_{}_{}".format(
                network_graph.nx_graph.number_of_nodes(),
                network_graph.nx_graph.number_of_edges(),
                number_of_service,
                number_of_demands_per_service,
                graph_name,
            )

            service_package = make_services(
                network_graph,
                number_of_demands_per_service,
                number_of_service,
                instance_name,
            )

            problem_instance = ProblemInstance(
                network_graph, service_package, instance_name
            )

            saved_file_path = os.path.join(instance_path, instance_name + ".pkl")

            pickle_save(problem_instance, saved_file_path)
