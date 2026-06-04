import time

import numpy as np

from core.generic_solver import GenericSolver
from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.solving_instance import SolvingInstance


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        max_pcc: int = 10,
        sigma: float = 2.0,
        max_iter: int = 200,
        max_evals: int | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        super().__init__(problem_instance)
        self.nx_graph = self.problem_instance.network_graph.nx_graph
        if max_pcc < 1:
            raise ValueError("max_pcc must be positive.")
        self.max_pcc = max_pcc
        self.sigma = sigma
        self.max_iter = max_iter
        self.max_evals = max_evals
        self.seed = seed
        self.verbose = verbose
        self.solver_name = "CMA-ES"

    def run(self):
        start_time = time.time()
        self.sol_ins = SolvingInstance(self.problem_instance)
        self.best_score = -np.inf
        self.best_solution = None
        x0 = self.initial_value()
        self.objective(x0)

        import cma

        opts = {
            "bounds": self._bounds(),
            "maxiter": self.max_iter,
            "seed": self.seed,
            "verbose": 0 if self.verbose else -9,
            "verb_disp": 1 if self.verbose else 0,
            "verb_log": 1 if self.verbose else 0,
        }
        if self.max_evals is not None:
            opts["maxfevals"] = self.max_evals

        es = cma.CMAEvolutionStrategy(x0, self.sigma, opts)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.objective(x) for x in solutions])
            if self.verbose:
                es.logger.add()
                es.disp()

        if self.best_solution is None:
            raise RuntimeError("CMA-ES did not find a feasible solution.")

        self.placement, self.routing, self.quantities = self.best_solution
        self.x_var, self.y_var, self.q_var = self.sol_ins.solution_to_arc_variables(
            self.placement,
            self.routing,
            self.quantities,
        )

        self.runtime = time.time() - start_time
        self.score = self.best_score
        self.build_time = 0.0
        return self

    def objective(self, value: np.ndarray):
        placement, routing, quantities = self.decode(value)
        if self.sol_ins.test_solution(placement, routing, quantities):
            score = sum(quantities)
            if score > self.best_score:
                self.best_score = score
                self.best_solution = (placement, routing, quantities)
            return -score

        flow_cap = sum(self.problem_instance.max_flows) + 1.0
        return (
            flow_cap
            + self.violation(placement, routing, quantities)
            - sum(quantities) / flow_cap
        )

    def decode(self, value: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands
        z = np.rint(value).astype(int)

        placement = np.clip(z[:S], 0, self.nx_graph.number_of_nodes() - 1)
        routing = np.clip(z[S : S + D], 0, self.max_pcc - 1)
        quantities = np.clip(
            z[-D:],
            self.problem_instance.min_flows,
            self.problem_instance.max_flows,
        )
        return placement.tolist(), routing.tolist(), quantities.tolist()

    def violation(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> float:
        graph = self.sol_ins.graph.copy()
        graph = self.sol_ins.initialize_graph_attributes(graph)
        graph = self.sol_ins.update_graph_attributes(
            graph,
            placement,
            routing,
            quantities,
        )

        capacity = sum(
            max(0.0, -graph.edges[e]["remaining_capacity"]) for e in graph.edges
        )
        resources = sum(
            max(0.0, -graph.nodes[v]["remaining_resources"]) for v in graph.nodes
        )
        latency = 0.0
        for d in range(len(quantities)):
            service = self.problem_instance.demand_to_service[d]
            path = self.sol_ins.compute_nth_shortest_path(
                self.problem_instance.origins[d],
                placement[service],
                routing[d],
            )
            delay = sum(
                graph.edges[e]["latency"] for e in path_node_to_path_tuple(path)
            )
            latency += max(0.0, delay - self.problem_instance.latencies[d])

        return capacity + resources + latency

    def initial_value(self, method: str = "heuristic") -> np.ndarray:
        if method == "basic":
            placement, routing, quantities = self.sol_ins.basic_placement_and_routing()
        elif method == "heuristic":
            from solvers.heuristic.heuristic import Solver as HeuristicSolver

            sol = HeuristicSolver(self.problem_instance)
            sol.run()
            placement, routing, quantities = sol.placement, sol.routing, sol.quantities
        else:
            raise ValueError("method must be 'basic' or 'heuristic'.")

        value = np.asarray(placement + routing + quantities, dtype=float)
        placement, routing, quantities = self.decode(value)
        return np.asarray(placement + routing + quantities, dtype=float)

    def _bounds(self) -> list[list[float]]:
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands
        n = self.nx_graph.number_of_nodes()
        lower = [0] * S + [0] * D + list(self.problem_instance.min_flows)
        upper = (
            [n - 1] * S
            + [self.max_pcc - 1] * D
            + list(self.problem_instance.max_flows)
        )
        return [lower, upper]
