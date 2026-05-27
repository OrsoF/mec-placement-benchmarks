from pathlib import Path

import networkx as nx
import numpy as np

from core.graph_utils import make_graph_from_gml
from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage
from data_io.experiment_io import pickle_save


rng = np.random.default_rng(0)


def edge_tier(edge: tuple[int, int], n_core: int) -> int:
    u, v = edge
    if u < n_core and v < n_core:
        return 0
    if u < n_core or v < n_core:
        return 1
    return 2


def make_three_level_graph(
    base_graph: nx.Graph,
    edge_per_core: int = 10,
    users_per_core: int = 20,
    user_degree: int = 4,
) -> tuple[NetworkGraph, int, np.ndarray]:
    # Three-tier synthetic edge-computing topology: imported backbone nodes
    # form the core, each edge node attaches to one random core node, and
    # each user attaches to a small random set of edge nodes.
    g = nx.convert_node_labels_to_integers(base_graph.copy())
    n_core = g.number_of_nodes()

    edge_nodes = np.arange(n_core, n_core * (1 + edge_per_core))
    user_nodes = np.arange(
        edge_nodes[-1] + 1,
        edge_nodes[-1] + 1 + n_core * users_per_core,
    )

    g.add_nodes_from(edge_nodes)
    g.add_nodes_from(user_nodes)

    core_parent = rng.integers(0, n_core, size=edge_nodes.size)
    g.add_edges_from(zip(edge_nodes, core_parent))

    edge_parent = rng.integers(
        edge_nodes[0],
        edge_nodes[-1] + 1,
        size=(user_nodes.size, user_degree),
    )
    g.add_edges_from((u, e) for u, row in zip(user_nodes, edge_parent) for e in row)

    resources = np.r_[
        np.full(n_core, 1_000_000),
        np.full(edge_nodes.size, 10_000),
        np.full(user_nodes.size, 100),
    ]

    tier_params = np.array(
        [
            [5000.0, 0.5, 5.0],
            [3000.0, 0.7, 7.0],
            [500.0, 1.5, 15.0],
        ]
    )
    tiers = np.fromiter((edge_tier(e, n_core) for e in g.edges), dtype=int)
    cap = tier_params[tiers, 0]
    alpha = tier_params[tiers, 1]
    beta = tier_params[tiers, 2]

    graph = NetworkGraph(
        g,
        resources.tolist(),
        cap.tolist(),
        alpha.tolist(),
        beta.tolist(),
    )
    return graph, n_core, user_nodes


def latency_budget(
    network_graph: NetworkGraph,
    n_services: int,
    scale: float = 25.0,
) -> float:
    g = network_graph.nx_graph
    alpha = np.fromiter((g.edges[e]["alpha"] for e in g.edges), dtype=float)
    beta = np.fromiter((g.edges[e]["beta"] for e in g.edges), dtype=float)
    return scale * (
        alpha.mean() * 5.0 * n_services**2 / g.number_of_edges() + beta.mean()
    )


def make_services(
    network_graph: NetworkGraph,
    n_services: int,
    n_core: int,
    user_nodes: np.ndarray,
    flow: float = 100.0,
) -> list[Service]:
    origins = np.r_[np.arange(n_core), user_nodes]
    latency = latency_budget(network_graph, n_services)
    resources = 5000 // n_services

    services = []
    for _ in range(n_services):
        demands = [
            Demand(
                origin_node_index=int(rng.choice(origins)),
                flow_min=1.0,
                flow_max=flow,
                max_latency=latency,
            )
            for _ in range(n_services)
        ]
        services.append(Service(demands, network_graph, resources))
    return services


def make_large_instance(
    gml_path: str | Path,
    n_services: int = 5,
    edge_per_core: int = 10,
    users_per_core: int = 20,
    user_degree: int = 4,
) -> ProblemInstance:
    gml_path = Path(gml_path)
    base = make_graph_from_gml(gml_path)
    graph, n_core, user_nodes = make_three_level_graph(
        base.nx_graph,
        edge_per_core=edge_per_core,
        users_per_core=users_per_core,
        user_degree=user_degree,
    )

    name = f"three_levels_{gml_path.stem}_{n_services}_{n_services}"
    services = make_services(graph, n_services, n_core, user_nodes)
    package = ServicePackage(services, name)
    return ProblemInstance(graph, package, name)


def save_large_instances(
    gml_dir: str | Path = Path("data") / "topologies",
    out_dir: str | Path = Path("artifacts") / "instances",
    n_services: tuple[int, ...] = (5,),
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gml_path in Path(gml_dir).glob("*.gml"):
        for n in n_services:
            instance = make_large_instance(gml_path, n_services=n)
            pickle_save(instance, out_dir / f"{instance.problem_name}.pkl")


def make_big_instances() -> None:
    save_large_instances()
