"""Construct a historical baseline candidate without claiming feasibility."""

import networkx as nx
import numpy as np

from mecbench.domain.contracts import ProblemData
from mecbench.domain.solution import Solution


def baseline_solution(problem: ProblemData) -> Solution:
    network = problem.network_graph
    graph = network.nx_graph
    placement_node = int(np.argmax(network.available_resources))
    x = np.zeros((problem.total_number_of_demands, graph.number_of_edges()))
    for d, origin in enumerate(problem.origins):
        nodes = nx.shortest_path(graph, origin, placement_node)
        for edge in zip(nodes, nodes[1:], strict=False):
            x[d, network.arc_index_dictionary[edge]] = 1.0
    y = np.zeros((problem.number_of_service, graph.number_of_nodes()))
    y[:, placement_node] = 1.0
    return Solution(x, y, np.asarray(problem.min_flows, dtype=float))
