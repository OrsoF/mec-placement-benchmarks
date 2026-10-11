import networkx as nx
import numpy as np

from mecbench.domain.network import NetworkGraph
from mecbench.domain.services import Demand, Service
from mecbench.instances.topology import build_large_graph


def three_level_graph(
    base_graph: nx.Graph,
    edge_per_core: int = 10,
    users_per_core: int = 2,
    user_degree: int = 1,
    rng: np.random.Generator | None = None,
):
    graph = build_large_graph(
        base_graph,
        edge_per_core=edge_per_core,
        users_per_edge=users_per_core,
        rng=rng,
    )
    node_resources = [
        1e2
        if graph.nodes[v].get("layer") == "user"
        else 1e4
        if graph.nodes[v].get("layer") == "edge"
        else 1e6
        for v in graph.nodes
    ]
    edge_capacity = [
        500
        if graph.nodes[u].get("layer") == "user" or graph.nodes[v].get("layer") == "user"
        else 3000
        for u, v in graph.edges
    ]
    network_graph = NetworkGraph(
        base_graph=graph,
        node_resources=node_resources,
        edge_capacity=edge_capacity,
        latency_alpha_factor=[0.5] * graph.number_of_edges(),
        latency_beta_factor=[5.0] * graph.number_of_edges(),
    )
    n_core = base_graph.number_of_nodes()
    user_nodes = [
        v
        for v in network_graph.nx_graph.nodes
        if network_graph.nx_graph.nodes[v].get("layer") == "user"
    ]
    return network_graph, n_core, np.asarray(user_nodes, dtype=int)


def random_services(
    network_graph: NetworkGraph,
    origin_nodes,
    n_services: int,
    demands_per_service: int,
    rng: np.random.Generator | None = None,
    latency_scale: float = 25.0,
):
    rng = rng or np.random.default_rng(0)
    graph = network_graph.nx_graph
    origins = list(origin_nodes)
    latency = latency_scale * (
        np.mean(network_graph.alphas) * n_services**2 / max(1, graph.number_of_edges())
        + np.mean(network_graph.betas)
    )
    services = []
    for _ in range(n_services):
        demands = [
            Demand(
                origin_node_index=int(rng.choice(origins)),
                flow_min=1,
                flow_max=100,
                max_latency=float(latency),
            )
            for _ in range(demands_per_service)
        ]
        services.append(
            Service(
                task_list=demands,
                network_graph=network_graph,
                resources_needed=max(1, 5000 // max(1, n_services)),
            )
        )
    return services
