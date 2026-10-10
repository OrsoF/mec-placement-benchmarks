import time

import numpy as np

from mecbench.solvers.base import GenericSolver
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.solution import SolvingInstance
from mecbench.domain.solution import clip_solution, greedy_quantity_lift, repair_solution


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        limit_shortest_path: int = 20,
        step_no_progression: int = 50,
        quantity_increment: int = 10,
        seed: int = 0,
        max_iter: int | None = None,
        patience: int | None = None,
    ):
        super().__init__(problem_instance)
        if limit_shortest_path < 1:
            raise ValueError("limit_shortest_path must be positive.")
        if quantity_increment < 1:
            raise ValueError("quantity_increment must be positive.")
        if step_no_progression < 1:
            raise ValueError("step_no_progression must be positive.")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive.")
        if max_iter is not None and max_iter < 1:
            raise ValueError("max_iter must be positive.")

        self.nx_graph = self.problem_instance.network_graph.nx_graph
        self.limit_shortest_path = limit_shortest_path
        self.quantity_increment = quantity_increment
        self.seed = seed
        self.patience = step_no_progression if patience is None else patience
        self.max_iter = max_iter or 50 * self.patience
        self.solver_name = "heuristic"
        self.solving_instance = SolvingInstance(self.problem_instance)

    def run(self):
        start = time.time()
        self.rng = np.random.default_rng(self.seed)

        placement, routing, quantities = self._initial_solution()
        best = (placement, routing, quantities)
        best_score = sum(quantities)
        current = best
        no_progress = 0
        self.visited_objectives = [best_score]

        for _ in range(self.max_iter):
            candidate = self._repair(*self._mutate(*current))
            if candidate is not None:
                current = candidate
                score = sum(candidate[2])
                if score > best_score:
                    best = candidate
                    best_score = score
                    no_progress = 0
                else:
                    no_progress += 1
            else:
                no_progress += 1

            self.visited_objectives.append(best_score)
            if no_progress >= self.patience:
                break

        self.placement, self.routing, self.quantities = best
        self.x_var, self.y_var, self.q_var = (
            self.solving_instance.solution_to_arc_variables(*best)
        )
        self.score = best_score
        self.runtime = time.time() - start
        self.build_time = 0.0
        return self

    def _initial_solution(self) -> tuple[list[int], list[int], list[int]]:
        placement, routing, quantities = (
            self.solving_instance.basic_placement_and_routing()
        )
        return self._maximize_quantities(placement, routing, quantities)

    def _mutate(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        p, r, q = placement.copy(), routing.copy(), quantities.copy()
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands

        op = self.rng.integers(3)
        if op == 0:
            p[self.rng.integers(S)] = self.rng.integers(self.nx_graph.number_of_nodes())
            q = list(self.problem_instance.min_flows)
        elif op == 1:
            r[self.rng.integers(D)] = self.rng.integers(self.limit_shortest_path)
            q = list(self.problem_instance.min_flows)
        else:
            d = self.rng.integers(D)
            q[d] += self.rng.choice((-self.quantity_increment, self.quantity_increment))

        return p, r, q

    def _repair(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]] | None:
        repaired = repair_solution(
            self.problem_instance,
            self.nx_graph,
            self.solving_instance,
            self.limit_shortest_path,
            placement,
            routing,
            quantities,
        )
        if repaired is None:
            return None
        return self._maximize_quantities(*repaired)

    def _maximize_quantities(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        D = self.problem_instance.total_number_of_demands
        max_steps = D * max(1, max(self.problem_instance.max_flows))
        return greedy_quantity_lift(
            self.problem_instance,
            self.solving_instance,
            self.rng,
            placement,
            routing,
            quantities,
            max_steps,
            self.quantity_increment,
        )

    def _clip(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        return clip_solution(
            self.problem_instance,
            self.nx_graph,
            self.limit_shortest_path,
            placement,
            routing,
            quantities,
        )
