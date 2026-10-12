import time

import networkx as nx
import numpy as np

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.solver_base import GenericSolver
from mecbench.utils.solver_workspace import SolvingInstance


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        limit_shortest_path: int = 4,
        candidate_nodes: int = 8,
        max_iter: int = 60,
        patience: int = 8,
        seed: int = 0,
    ):
        super().__init__(problem_instance)
        self.solver_name = "fast_heuristic"
        self.limit_shortest_path = int(limit_shortest_path)
        self.candidate_nodes = int(candidate_nodes)
        self.max_iter = int(max_iter)
        self.patience = int(patience)
        self.seed = seed

        self.solving_instance = SolvingInstance(problem_instance)
        self.g = problem_instance.network_graph.nx_graph
        self.n = self.g.number_of_nodes()
        self.m = self.g.number_of_edges()
        self.S = problem_instance.number_of_service
        self.D = problem_instance.total_number_of_demands

        self.service = np.asarray(problem_instance.demand_to_service, dtype=int)
        self.orig = np.asarray(problem_instance.origins, dtype=int)
        self.qmin = np.asarray(problem_instance.min_flows, dtype=float)
        self.qmax = np.asarray(problem_instance.max_flows, dtype=float)
        self.lat = np.asarray(problem_instance.latencies, dtype=float)
        self.res = np.asarray(problem_instance.needed_resources, dtype=float)
        self.node_cap = np.asarray(problem_instance.network_graph.available_resources, dtype=float)
        self.edge_cap = np.asarray(problem_instance.network_graph.capacities, dtype=float)
        self.alpha = np.asarray(problem_instance.network_graph.alphas, dtype=float)
        self.beta = np.asarray(problem_instance.network_graph.betas, dtype=float)
        self.arc_index = problem_instance.network_graph.arc_index_dictionary
        self._path_cache = {}
        self._dist = None
        self._service_nodes = None

    def run(self):
        start = time.time()
        self.rng = np.random.default_rng(self.seed)
        self._dist = dict(nx.all_pairs_shortest_path_length(nx.Graph(self.g)))
        self._service_nodes = self._candidate_nodes()

        p = self._initial_placement()
        r = np.zeros(self.D, dtype=int)
        q = self._lift(p, r, self.qmin.copy())
        best = (p.copy(), r.copy(), q.copy())
        current = best
        best_score = float(q.sum())
        upper = float(self.qmax.sum())
        no_progress = 0
        self.visited_objectives = [best_score]

        for _ in range(self.max_iter):
            if best_score >= upper:
                break
            cand = self._best_neighbor(*current)
            if cand is None:
                cand = self._perturb(*current)

            score = float(cand[2].sum())
            current = cand
            if score > best_score:
                best = tuple(x.copy() for x in cand)
                best_score = score
                no_progress = 0
            else:
                no_progress += 1

            self.visited_objectives.append(best_score)
            if no_progress >= self.patience:
                break

        self.placement = best[0].astype(int).tolist()
        self.routing = best[1].astype(int).tolist()
        self.quantities = best[2].tolist()
        self.x_var, self.y_var, self.q_var = self.solving_instance.solution_to_arc_variables(
            self.placement,
            self.routing,
            self.quantities,
        )
        self.score = best_score
        self.runtime = time.time() - start
        self.build_time = 0.0
        return self

    def _candidate_nodes(self) -> list[np.ndarray]:
        top = np.argsort(-self.node_cap)[: self.candidate_nodes]
        nodes = []
        for s in range(self.S):
            d = np.flatnonzero(self.service == s)
            origins = np.unique(self.orig[d])
            score = np.zeros(self.n)
            for u in origins:
                du = self._dist[int(u)]
                score += np.fromiter((du.get(v, self.n) for v in range(self.n)), float, self.n)
            close = np.argsort(score)[: self.candidate_nodes]
            cand = np.unique(np.concatenate((origins, top, close)))
            cand = cand[self.node_cap[cand] >= self.res[s]]
            nodes.append(cand.astype(int))
        return nodes

    def _initial_placement(self) -> np.ndarray:
        p = np.empty(self.S, dtype=int)
        rem = self.node_cap.copy()
        order = np.argsort(-self.res)
        for s in order:
            d = np.flatnonzero(self.service == s)
            best, best_score = None, np.inf
            for v in self._service_nodes[s]:
                if rem[v] < self.res[s]:
                    continue
                dist = sum(
                    self.qmax[j] * self._dist[int(self.orig[j])].get(int(v), self.n) for j in d
                )
                score = dist - 1e-6 * rem[v]
                if score < best_score:
                    best, best_score = int(v), score
            if best is None:
                best = int(np.argmax(rem))
            p[s] = best
            rem[best] -= self.res[s]
        return p

    def _path(self, source: int, dest: int, k: int):
        key = (int(source), int(dest), int(k))
        if key in self._path_cache:
            return self._path_cache[key]

        path = self.solving_instance.compute_nth_shortest_path(source, dest, k)
        idx = np.fromiter(
            (self.arc_index[(path[i], path[i + 1])] for i in range(len(path) - 1)),
            dtype=int,
            count=max(0, len(path) - 1),
        )
        self._path_cache[key] = idx
        return idx

    def _paths(self, p, r):
        paths = []
        for d in range(self.D):
            paths.append(self._path(self.orig[d], p[self.service[d]], r[d]))
        return paths

    def _loads(self, p, r, q):
        paths = self._paths(p, r)
        edge_load = np.zeros(self.m)
        for d, idx in enumerate(paths):
            edge_load[idx] += q[d]
        node_load = np.bincount(p, weights=self.res, minlength=self.n)
        return paths, edge_load, node_load

    def _feasible(self, p, r, q) -> bool:
        return self.solving_instance.test_compact_solution(p, r, q)

    def _lift(self, p, r, q):
        q = np.minimum(np.maximum(q, self.qmin), self.qmax).astype(float)
        if not self._feasible(p, r, q):
            q = self.qmin.copy()
            if not self._feasible(p, r, q):
                return q

        paths = self._paths(p, r)
        incidence = np.zeros((self.D, self.m))
        for d, idx in enumerate(paths):
            incidence[d, idx] = 1
        # Entry (i, d) is the latency increase for demand i when q[d]
        # increases by one, including every arc shared by the two routes.
        latency_sensitivity = (incidence * self.alpha) @ incidence.T

        while True:
            paths, edge_load, _ = self._loads(p, r, q)
            latency_slack = self.lat - incidence @ (self.beta + self.alpha * edge_load)
            room = np.zeros(self.D)
            for d, idx in enumerate(paths):
                flow_room = self.qmax[d] - q[d]
                if flow_room <= 0:
                    continue
                cap_room = np.inf if len(idx) == 0 else np.min(self.edge_cap[idx] - edge_load[idx])
                affected = latency_sensitivity[:, d] > 0
                lat_room = (
                    np.min(latency_slack[affected] / latency_sensitivity[affected, d])
                    if np.any(affected)
                    else np.inf
                )
                room[d] = max(0.0, min(flow_room, cap_room, lat_room))

            d = int(np.argmax(room))
            delta = np.floor(room[d])
            if delta <= 0:
                break
            q[d] += delta
            if not self._feasible(p, r, q):
                q[d] -= delta
                break
        return q

    def _repair_lift(self, p, r, q):
        p = np.asarray(p, dtype=int).copy()
        r = np.clip(np.asarray(r, dtype=int), 0, self.limit_shortest_path - 1)
        q = np.asarray(q, dtype=float).copy()
        if not self._feasible(p, r, q):
            q = self.qmin.copy()
        if not self._feasible(p, r, q):
            return None
        return p, r, self._lift(p, r, q)

    def _best_neighbor(self, p, r, q):
        best_score = float(q.sum())

        for s in self.rng.permutation(self.S)[: min(self.S, 8)]:
            old = p[s]
            nodes = self.rng.permutation(self._service_nodes[s])[
                : min(len(self._service_nodes[s]), 6)
            ]
            for v in nodes:
                if v == old:
                    continue
                pp = p.copy()
                pp[s] = v
                cand = self._repair_lift(pp, r, self.qmin)
                if cand is None:
                    continue
                score = float(cand[2].sum())
                if score > best_score:
                    return cand

        for d in self.rng.permutation(self.D)[: min(self.D, 16)]:
            old = r[d]
            for k in range(self.limit_shortest_path):
                if k == old:
                    continue
                rr = r.copy()
                rr[d] = k
                cand = self._repair_lift(p, rr, q)
                if cand is None:
                    cand = self._repair_lift(p, rr, self.qmin)
                if cand is None:
                    continue
                score = float(cand[2].sum())
                if score > best_score:
                    return cand

        return None

    def _perturb(self, p, r, q):
        pp, rr = p.copy(), r.copy()
        if self.rng.random() < 0.5:
            s = int(self.rng.integers(self.S))
            cand = self._service_nodes[s]
            if len(cand):
                pp[s] = int(self.rng.choice(cand))
        else:
            d = int(self.rng.integers(self.D))
            rr[d] = int(self.rng.integers(self.limit_shortest_path))

        cand = self._repair_lift(pp, rr, self.qmin)
        if cand is None:
            return p.copy(), r.copy(), q.copy()
        return cand
