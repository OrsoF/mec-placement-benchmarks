from problem_classes.networkgraph import NetworkGraph
from networkx.readwrite.gml import read_gml
from collections import defaultdict
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


def get_outgoing_arc(nx_graph: nx.DiGraph, node: int) -> list[tuple]:
    return [(node, elem) for elem in nx_graph.neighbors(node)]


def get_encoded_outgoing_arc(nx_graph: nx.DiGraph, arc_dic: dict, node):
    return [arc_dic[elem] for elem in get_outgoing_arc(nx_graph, node)]


def get_incoming_arc(nx_graph: nx.DiGraph, node) -> list[tuple]:
    """
    Returns all arcs that get into the node "node".
    """
    return [(neighbor_node, node) for neighbor_node in nx_graph.neighbors(node)]


def get_encoded_incoming_arc(
    nx_graph: nx.DiGraph, arc_dic: dict, node: int
) -> list[int]:
    """
    Returns all arcs that get into the node "node"
    and encode each arc as an integer take from arc_dic.
    """
    result = []
    for elem in get_incoming_arc(nx_graph, node):
        try:
            result.append(arc_dic[elem])
        except AttributeError:
            assert False

    return [arc_dic[elem] for elem in get_incoming_arc(nx_graph, node)]


def edges_to_path(edges: list):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    start = next(node for node in adj if len(adj[node]) == 1)
    path, visited, current = [start], set(), start
    while True:
        next_node = next((n for n in adj[current] if (current, n) not in visited), None)
        if not next_node:
            break
        visited.add(
            (current, next_node) if current < next_node else (next_node, current)
        )
        path.append(next_node)
        current = next_node
    return path
