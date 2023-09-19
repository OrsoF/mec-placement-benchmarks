import networkx as nx


def check_network_graph(nx_graph: nx.DiGraph):
    assert isinstance(
        nx_graph, nx.DiGraph
    ), "Considered graph is not a DiGraph instance."
    assert is_graph_symmetric(nx_graph), "Considered graph is not symmetric."


def is_graph_symmetric(nx_graph: nx.DiGraph) -> bool:
    """
    Check if a given DiGraph is symmetric.
    """
    for edge in nx_graph.edges:
        if (edge[1], edge[0]) not in nx_graph.edges:
            return False
    return True
