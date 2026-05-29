from pathlib import Path

import networkx as nx
import numpy as np

from core.graph_utils import make_graph_from_gml
from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage


def instance(graph: NetworkGraph, services: list[Service], name: str) -> ProblemInstance:
    return ProblemInstance(graph, ServicePackage(services, name), name)


def user_cloud_services(
    graph: NetworkGraph,
    users: np.ndarray,
    cloud: int,
    latency: float,
) -> list[Service]:
    return [
        Service(
            [
                Demand(int(u), 1.0, 5.0, latency),
                Demand(int(cloud), 1.0, 2.0, latency),
            ],
            graph,
            resources_needed=1,
        )
        for u in users
    ]


def latency_budget(graph: NetworkGraph, n_services: int, scale: float) -> float:
    g = graph.nx_graph
    alpha = np.fromiter((g.edges[e]["alpha"] for e in g.edges), dtype=float)
    beta = np.fromiter((g.edges[e]["beta"] for e in g.edges), dtype=float)
    return scale * (
        alpha.mean() * 5.0 * n_services**2 / g.number_of_edges() + beta.mean()
    )


def random_services(
    graph: NetworkGraph,
    origins: np.ndarray,
    n_services: int,
    demands_per_service: int,
    rng: np.random.Generator,
    flow: float = 100.0,
    latency_scale: float = 7.0,
) -> list[Service]:
    latency = latency_budget(graph, n_services, latency_scale)
    resources = 5000 // n_services
    return [
        Service(
            [
                Demand(int(u), 1.0, flow, latency)
                for u in rng.choice(origins, size=demands_per_service)
            ],
            graph,
            resources,
        )
        for _ in range(n_services)
    ]


def line_edges(nodes: list[str], r: int) -> list[tuple[str, str]]:
    return [
        (u, nodes[j])
        for i, u in enumerate(nodes)
        for j in range(max(0, i - r), min(len(nodes), i + r + 1))
        if i != j
    ]


def ring_edges(nodes: list[str], r: int) -> list[tuple[str, str]]:
    return [
        (u, nodes[(i + d) % len(nodes)])
        for i, u in enumerate(nodes)
        for d in range(-r, r + 1)
        if d
    ]


def graph_from_gml(path: str | Path) -> NetworkGraph:
    return make_graph_from_gml(str(path))


def edge_tier(edge: tuple[int, int], n_core: int) -> int:
    u, v = edge
    if u < n_core and v < n_core:
        return 0
    if u < n_core or v < n_core:
        return 1
    return 2


def three_level_graph(
    base_graph: nx.Graph,
    edge_per_core: int,
    users_per_core: int,
    user_degree: int,
    rng: np.random.Generator,
) -> tuple[NetworkGraph, int, np.ndarray]:
    assert edge_per_core >= 1
    assert users_per_core >= 1
    assert user_degree >= 1

    g = nx.convert_node_labels_to_integers(nx.Graph(base_graph))
    n_core = g.number_of_nodes()
    edge_nodes = np.arange(n_core, n_core * (1 + edge_per_core))
    user_nodes = np.arange(
        edge_nodes[-1] + 1,
        edge_nodes[-1] + 1 + n_core * users_per_core,
    )

    g.add_nodes_from(edge_nodes)
    g.add_nodes_from(user_nodes)
    g.add_edges_from(zip(edge_nodes, rng.integers(0, n_core, size=edge_nodes.size)))
    parent = rng.integers(
        edge_nodes[0],
        edge_nodes[-1] + 1,
        size=(user_nodes.size, user_degree),
    )
    g.add_edges_from((u, e) for u, row in zip(user_nodes, parent) for e in row)

    resources = np.r_[
        np.full(n_core, 1_000_000.0),
        np.full(edge_nodes.size, 10_000.0),
        np.full(user_nodes.size, 100.0),
    ]
    params = np.array([[5000.0, 0.5, 5.0], [3000.0, 0.7, 7.0], [500.0, 1.5, 15.0]])
    edge_params = params[
        np.fromiter((edge_tier(e, n_core) for e in g.edges), dtype=int)
    ]
    graph = NetworkGraph(
        g,
        resources.tolist(),
        edge_params[:, 0].tolist(),
        edge_params[:, 1].tolist(),
        edge_params[:, 2].tolist(),
    )
    return graph, n_core, user_nodes
