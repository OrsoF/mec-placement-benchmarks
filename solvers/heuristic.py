"""
Améliorer:
- Augmentation greedy de q
- 
"""

import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from problem_classes.service import Service
from itertools import islice
from problem_classes.probleminstance import ProblemInstance
from utils.generic_solver import GenericSolver
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node
from problem_classes.solvinginstance import SolvingInstance

rnd_gen = np.random.default_rng(0)


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        limit_shortest_path: int = 2,
        step_no_progression: int = 5,
    ):
        self.problem_instance = problem_instance
        self.limit_shortest_path = limit_shortest_path
        self.step_no_progression = step_no_progression

        self.nx_graph: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "heuristic"
        self.solving_instance = SolvingInstance(self.problem_instance)

    def run(
        self,
    ):
        start_time = time.time()

        self.visited_objectives = []

        placement, routing, quantities = (
            self.solving_instance.basic_placement_and_routing()
        )
        placement, routing, quantities = self.solving_instance.maximize_quantities(
            placement, routing, quantities
        )
        self.visited_objectives.append(sum(quantities))

        while self.continue_search():
            # print(self.visited_objectives[-1])
            new_placement = placement.copy()
            new_placement[rnd_gen.integers(self.problem_instance.number_of_service)] = (
                rnd_gen.integers(self.nx_graph.number_of_nodes())
            )
            new_routing = routing.copy()
            new_routing[
                rnd_gen.integers(self.problem_instance.total_number_of_demands)
            ] = self.limit_shortest_path
            new_quantities = [1] * self.problem_instance.total_number_of_demands
            if self.solving_instance.test_solution(
                new_placement, new_routing, new_quantities
            ):
                placement = new_placement
                routing = new_routing
                placement, routing, quantities = (
                    self.solving_instance.maximize_quantities(
                        placement, routing, new_quantities
                    )
                )

                if sum(quantities) > self.visited_objectives[-1]:
                    self.visited_objectives.append(sum(quantities))
                else:
                    self.visited_objectives.append(self.visited_objectives[-1])

        self.score = max(self.visited_objectives)
        # print(f"Heuristic score: {self.score}")
        self.runtime = time.time() - start_time
        # print(placement, routing, quantities)
        self.placement = placement
        self.routing = routing
        self.quantities = quantities

    def continue_search(self) -> bool:
        return len(self.visited_objectives) < self.step_no_progression or (
            min(self.visited_objectives[-self.step_no_progression :])
            != max(self.visited_objectives[-self.step_no_progression :])
        )
