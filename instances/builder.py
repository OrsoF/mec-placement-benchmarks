from pathlib import Path

import networkx as nx
import numpy as np

from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from instances.utility import (
    graph_from_gml,
    instance,
    line_edges,
    random_services,
    ring_edges,
    three_level_graph,
    user_cloud_services,
)


def make_tree_instance(
    number_of_users: int = 7,
    neighbors_connection: int = 2,
    randomize: bool = False,
    rng: np.random.Generator | None = None,
) -> ProblemInstance:
    assert number_of_users >= 1
    assert neighbors_connection >= 0

    users = [f"user_{i}" for i in range(number_of_users)]
    wap, cloud = "wap", "cloud"

    g = nx.Graph()
    g.add_nodes_from(users + [wap, cloud])
    g.add_edges_from(line_edges(users, neighbors_connection))
    g.add_edges_from((wap, u) for u in users)
    g.add_edge(wap, cloud)

    cap = np.array(
        [
            3 * number_of_users
            if {u, v} == {wap, cloud}
            else 50
            if wap in {u, v}
            else 10 * number_of_users
            for u, v in g.edges
        ],
        dtype=float,
    )
    resources = np.r_[np.full(number_of_users, 10.0), 50.0, 1000.0]
    if randomize:
        rng = np.random.default_rng(0) if rng is None else rng
        resources[:number_of_users] = rng.integers(5, 16, size=number_of_users)

    graph = NetworkGraph(
        g,
        resources.tolist(),
        cap.tolist(),
        [3.0] * len(cap),
        [3.0] * len(cap),
    )
    services = user_cloud_services(
        graph,
        np.arange(number_of_users),
        number_of_users + 1,
        10.0 * number_of_users,
    )
    name = f"tree_{number_of_users}_{neighbors_connection}"
    return instance(graph, services, name)


def make_star_instance(
    number_of_groups: int,
    users_per_group: int,
    neighbors_connection: int,
    rng: np.random.Generator | None = None,
) -> ProblemInstance:
    assert number_of_groups >= 1
    assert users_per_group >= 1
    assert neighbors_connection >= 0
    assert neighbors_connection <= users_per_group // 2, "Too many neighbors."

    rng = np.random.default_rng(0) if rng is None else rng
    users = [
        f"user_{group}_{user}"
        for group in range(number_of_groups)
        for user in range(users_per_group)
    ]
    waps = [f"wap_{group}" for group in range(number_of_groups)]
    cloud = "cloud"

    g = nx.Graph()
    g.add_nodes_from(users + waps + [cloud])
    for group, wap in enumerate(waps):
        group_users = users[group * users_per_group : (group + 1) * users_per_group]
        g.add_edges_from(ring_edges(group_users, neighbors_connection))
        g.add_edges_from((u, wap) for u in group_users)
        g.add_edge(wap, cloud)

    n_users = number_of_groups * users_per_group
    resources = np.r_[
        rng.integers(0, 2, size=n_users),
        rng.integers(0, users_per_group + 1, size=number_of_groups),
        n_users,
    ].astype(float)
    cap = np.array(
        [
            n_users
            if cloud in {u, v}
            else users_per_group
            if any(x.startswith("wap_") for x in (u, v))
            else users_per_group
            for u, v in g.edges
        ],
        dtype=float,
    )

    graph = NetworkGraph(
        g,
        resources.tolist(),
        cap.tolist(),
        [3.0] * len(cap),
        [3.0] * len(cap),
    )
    services = user_cloud_services(
        graph,
        np.arange(n_users),
        n_users + number_of_groups,
        10.0 * n_users,
    )
    name = f"star_{number_of_groups}_{users_per_group}_{neighbors_connection}"
    return instance(graph, services, name)


def make_large_instance(
    gml_path: str | Path,
    n_services: int = 5,
    edge_per_core: int = 10,
    users_per_core: int = 20,
    user_degree: int = 4,
    rng: np.random.Generator | None = None,
) -> ProblemInstance:
    rng = np.random.default_rng(0) if rng is None else rng
    gml_path = Path(gml_path)
    graph, n_core, user_nodes = three_level_graph(
        graph_from_gml(gml_path).nx_graph,
        edge_per_core,
        users_per_core,
        user_degree,
        rng,
    )
    services = random_services(
        graph,
        np.r_[np.arange(n_core), user_nodes],
        n_services,
        n_services,
        rng,
        latency_scale=25.0,
    )
    name = f"three_levels_{gml_path.stem}_{n_services}_{n_services}"
    return instance(graph, services, name)
