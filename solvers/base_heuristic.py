import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from problem_classes.service import Service
from itertools import islice
from problem_classes.probleminstance import ProblemInstance
from utils.instance_solver import InstanceSolver


class Solver(InstanceSolver):
    def __init__(self, problem_instance: ProblemInstance, limit_shortest_path: int = 1):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.service_package = self.problem_instance.service_package

        self.network_graph = self.problem_instance.network_graph
        self.graph: nx.DiGraph = self.network_graph.nx_graph

        self.limit_shortest_path = limit_shortest_path
        self.solver_name = "base_heuristic"

        self.service_node_matrix: np.ndarray  # y matrix
        self.quantity_matrix: np.ndarray  # q matrix
        self.demand_arc_transit: np.ndarray  # x matrix

        # Useful quantities
        self.service_placement = np.random.randint(
            self.graph.number_of_nodes(), size=self.problem_instance.number_of_service
        )  # Where is placed each service
        self.sp_indices = np.zeros(
            (self.problem_instance.total_number_of_demands)
        )  # The shortest path indice for each demand
        self.choosen_quantities = np.ones(
            (self.problem_instance.total_number_of_demands)
        )  # We chose one as quantity for each demand
        self.paths_used = [
            self._select_shortest_path(
                self.problem_instance.origins[demand_index],
                self.service_placement[
                    self.problem_instance.demand_to_service[demand_index]
                ],
                0,
            )
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]

        self._test_solution()

    def _update_edge_attributes(self):
        # Sur chaque Edge : quantité, latence, remaining quantity
        for demand_index in range(self.problem_instance.total_number_of_demands):
            for node_index in range(len(self.paths_used[demand_index]) - 1):
                edge = (
                    self.paths_used[demand_index][node_index],
                    self.paths_used[demand_index][node_index + 1],
                )
                # Update Q
                self.graph.edges[edge]["quantity"] += self.choosen_quantities[
                    demand_index
                ]
                # Update latence
                self.graph.edges[edge]["latency"] += (
                    self.choosen_quantities[demand_index]
                    * self.graph.edges[edge]["alpha"]
                )

        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = (
                self.graph.edges[edge]["capacity"] - self.graph.edges[edge]["quantity"]
            )

    def _reset_edge_attributes(self):
        for edge in self.graph.edges:
            # Reset Q, l
            self.graph.edges[edge]["quantity"] = 0
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]

    def _update_node_attributes(self):
        for service_index in range(self.problem_instance.number_of_service):
            target_node = self.service_placement[service_index]
            required_resources = self.problem_instance.needed_resources[service_index]
            self.graph.nodes[target_node]["remaining_resources"] -= required_resources

    def _reset_node_attributes(self):
        for node in self.graph.nodes:
            # Update remaining resources
            self.graph.nodes[node]["remaining_resources"] = self.graph.nodes[node][
                "available_resources"
            ]

    def _assert_all_constraints(self):
        # Contraintes :
        # - Capacité pour chaque arc
        # - Pour chaque demande, quantité et latence limite
        # - Ressource pour chaque noeud
        for edge in self.graph.edges:
            assert (
                self.graph.edges[edge]["remaining_capacity"] >= 0
            ), "Capacité non respectée."

        for demand_index in range(self.problem_instance.total_number_of_demands):
            assert (
                self.choosen_quantities[demand_index]
                <= self.problem_instance.max_flows[demand_index]
            ), "Max flow exceeded."
            latency = 0
            for node_index in range(len(self.paths_used[demand_index]) - 1):
                edge = (
                    self.paths_used[demand_index][node_index],
                    self.paths_used[demand_index][node_index + 1],
                )
                latency += self.graph.edges[edge]["latency"]
            assert (
                latency <= self.problem_instance.latencies[demand_index]
            ), "Latency exceeded."

        for node in self.graph.nodes:
            assert (
                self.graph.nodes[node]["remaining_resources"] >= 0
            ), "No enough resources."

    def _select_shortest_path(self, source: int, destination: int, index: int):
        index = int(index)
        shortest_paths = nx.algorithms.simple_paths.shortest_simple_paths(
            self.graph,
            source,
            destination,
        )
        for i, path in enumerate(shortest_paths):
            if i == index:
                return path
            last_visited_path = path
        return last_visited_path

    def run(
        self,
        options: dict = None,
        solving_time_limit: float = 600,
        thread_limit: int = 1,
        solver: str = "gurobi",
    ):
        start_time = time.time()

        # Résolution = (x, y, q)

        # Association noeuds/services
        self._services_nodes_association()

        # Calcul des shortest paths
        self._routing()

        # Augmentation des quantités
        self._update_quantities()

        self.runtime = time.time() - start_time
        objective_value = np.sum(self.choosen_quantities)

        print(objective_value)

        self.runtime = time.time() - start_time
        self.lower_bound = objective_value
        self.upper_bound = objective_value

    def _services_nodes_association(self):
        pass

    def _routing(self):
        pass

    def _update_quantities(self):

        max_iter = 1000
        for _ in range(max_iter):
            demand_index = np.random.randint(
                self.problem_instance.total_number_of_demands
            )
            self.choosen_quantities[demand_index] += 1
            try:
                self._test_solution()
            except:
                self.choosen_quantities[demand_index] -= 1

    def _test_solution(self):
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes()
        self._update_node_attributes()
        self._assert_all_constraints()
