import numpy as np
import networkx as nx

from core.graph_utils import path_node_to_path_tuple


def to_int_array(x) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=float)).astype(int)


def split_state(problem, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(s)
    S = problem.number_of_service
    D = problem.total_number_of_demands
    return s[:S], s[S : S + D], s[S + D : S + 2 * D]


def make_state(placement, routing, quantities) -> np.ndarray:
    return to_int_array(np.concatenate((placement, routing, quantities)))


def basic_solution(problem, graph: nx.Graph) -> tuple[list[int], list[int], list[int]]:
    resources = np.fromiter(
        (graph.nodes[v]["available_resources"] for v in graph.nodes),
        dtype=float,
        count=graph.number_of_nodes(),
    )
    cloud = int(np.argmax(resources))
    return (
        [cloud] * problem.number_of_service,
        [0] * problem.total_number_of_demands,
        list(problem.min_flows),
    )


def basic_state(problem, graph: nx.Graph) -> np.ndarray:
    return make_state(*basic_solution(problem, graph))


def clip_solution(problem, graph: nx.Graph, max_path, placement, routing, quantities):
    n = graph.number_of_nodes()
    p = np.clip(placement, 0, n - 1).astype(int).tolist()
    r = np.clip(routing, 0, max_path - 1).astype(int).tolist()
    q = np.clip(quantities, problem.min_flows, problem.max_flows).astype(int).tolist()
    return p, r, q


def clip_state(problem, graph: nx.Graph, max_path, s) -> np.ndarray:
    s = to_int_array(s)
    p, r, q = split_state(problem, s)
    p[:], r[:], q[:] = clip_solution(problem, graph, max_path, p, r, q)
    return s


def greedy_quantity_lift(problem, checker, rng, placement, routing, quantities, steps, increment):
    q = list(quantities)
    D = problem.total_number_of_demands
    for _ in range(steps):
        d = int(rng.integers(D))
        if q[d] >= problem.max_flows[d]:
            continue
        delta = min(increment, problem.max_flows[d] - q[d])
        q[d] += delta
        if not checker.test_solution(placement, routing, q):
            q[d] -= delta
    return placement, routing, q


def repair_solution(problem, graph, checker, max_path, placement, routing, quantities):
    p, r, q = clip_solution(problem, graph, max_path, placement, routing, quantities)
    if checker.test_solution(p, r, q):
        return p, r, q
    q = list(problem.min_flows)
    if checker.test_solution(p, r, q):
        return p, r, q
    return None


def validate_compact_shapes(problem, graph: nx.Graph, placement, routing, quantities):
    if len(placement) != problem.number_of_service:
        raise ValueError("placement length must match the number of services.")
    if len(routing) != problem.total_number_of_demands:
        raise ValueError("routing length must match the number of demands.")
    if len(quantities) != problem.total_number_of_demands:
        raise ValueError("quantities length must match the number of demands.")
    if not all(0 <= int(v) < graph.number_of_nodes() for v in placement):
        raise ValueError("placement contains an unknown node.")
    if not all(int(k) >= 0 for k in routing):
        raise ValueError("routing contains a negative path index.")


def compact_to_arc(problem, graph: nx.Graph, path_fn, placement, routing, quantities):
    validate_compact_shapes(problem, graph, placement, routing, quantities)

    y = np.zeros((problem.number_of_service, graph.number_of_nodes()), dtype=float)
    y[np.arange(problem.number_of_service), np.asarray(placement, dtype=int)] = 1.0

    x = np.zeros((problem.total_number_of_demands, graph.number_of_edges()), dtype=float)
    for d in range(problem.total_number_of_demands):
        k = problem.demand_to_service[d]
        path = path_fn(problem.origins[d], int(placement[k]), int(routing[d]))
        for e in path_node_to_path_tuple(path):
            x[d, problem.network_graph.arc_index_dictionary[e]] = 1.0

    return x, y, np.asarray(quantities, dtype=float)


def residual_arrays(problem, x, y, q):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    edge_load = x.T @ q
    node_load = y.T @ np.asarray(problem.needed_resources, dtype=float)
    capacities = np.asarray(problem.network_graph.capacities, dtype=float)
    resources = np.asarray(problem.network_graph.available_resources, dtype=float)
    return {
        "edge_load": edge_load,
        "node_load": node_load,
        "remaining_capacity": capacities - edge_load,
        "remaining_resources": resources - node_load,
    }


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


def validate_arc_solution(problem, x, y, q) -> None:
    graph = problem.network_graph.nx_graph
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    assert x.shape == (problem.total_number_of_demands, graph.number_of_edges())
    assert y.shape == (problem.number_of_service, graph.number_of_nodes())
    assert q.shape == (problem.total_number_of_demands,)

    assert ((y.sum(axis=1) - 1) ** 2).max() < 1e-2

    residual = residual_arrays(problem, x, y, q)
    assert np.all(residual["node_load"] <= np.asarray(problem.network_graph.available_resources))
    assert np.all(np.asarray(problem.min_flows) <= q), (
        f"Demand quantity below minimum flow : {q} >= {np.asarray(problem.min_flows)}"
    )
    assert np.all(q <= np.asarray(problem.max_flows)), (
        f"Demand quantity above maximum flow : {q} <= {np.asarray(problem.max_flows)}"
    )
    assert np.all(residual["edge_load"] <= np.asarray(problem.network_graph.capacities)), (
        f"Edge capacity exceeded : {residual['edge_load']} <= "
        f"{np.asarray(problem.network_graph.capacities)}"
    )

    for d in range(problem.total_number_of_demands):
        edge_idx = np.nonzero(x[d])[0]
        if len(edge_idx):
            k = problem.demand_to_service[d]
            origin = problem.origins[d]
            dest = int(np.argmax(y[k]))
            edges = [problem.network_graph.arc_list[a] for a in edge_idx]
            path = nx.dag_longest_path(nx.DiGraph(edges))
            assert path[0] == origin and path[-1] == dest

        latency = sum(
            problem.network_graph.alphas[a] * q[d] + problem.network_graph.betas[a]
            for a in edge_idx
        )
        assert latency <= problem.latencies[d]
