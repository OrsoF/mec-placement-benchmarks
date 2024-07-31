from problem_classes.networkgraph import NetworkGraph
from networkx.readwrite.gml import read_gml
import networkx as nx


def make_graph_from_gml(graph_path: str) -> NetworkGraph:
    """
    Read graph from gml file.
    Add resources, capacities, alphas and betas.
    """
    nx_graph: nx.Graph = read_gml(graph_path)
    resources = [10**4 for _ in nx_graph.nodes]
    capacities = [200 + 200 * len(nx_graph.nodes) // 15 for _ in nx_graph.edges] * 2
    alphas = [1 / 5.0 for _ in nx_graph.edges] * 2
    betas = [5.0 for _ in nx_graph.edges] * 2

    network_graph: NetworkGraph = NetworkGraph(
        nx_graph, resources, capacities, alphas, betas
    )

    return network_graph


def path_node_to_path_tuple(path: list) -> list:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_tuple_to_path_node(path: list) -> list:
    return [elem[0] for elem in path] + [path[-1][1]]
