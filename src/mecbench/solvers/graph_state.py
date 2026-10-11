"""Apply residuals to a mutable working graph."""

import networkx as nx

from mecbench.domain.loads import residual_arrays


def apply_residuals_to_graph(problem, graph: nx.Graph, x, y, q) -> nx.Graph:
    residual = residual_arrays(problem, x, y, q)
    for u in graph.nodes:
        graph.nodes[u]["remaining_resources"] = residual["remaining_resources"][u]

    for a, e in enumerate(problem.network_graph.arc_list):
        if not graph.has_edge(*e):
            continue
        load = residual["edge_load"][a]
        graph.edges[e]["remaining_capacity"] = residual["remaining_capacity"][a]
        graph.edges[e]["latency"] = graph.edges[e]["beta"] + load * graph.edges[e]["alpha"]
    return graph
