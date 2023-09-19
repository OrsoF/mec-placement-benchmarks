import networkx as nx
import numpy as np


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
        except:
            print(elem)
            print(node)
            print(nx_graph.neighbors(node))
            print((9, 15) in nx_graph.edges)
            print((15, 9) in nx_graph.edges)
            assert False

    return [arc_dic[elem] for elem in get_incoming_arc(nx_graph, node)]
