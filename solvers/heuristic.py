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
    def __init__(self, problem_instance: ProblemInstance, limit_shortest_path: int = 1):
        self.problem_instance = problem_instance
        self.limit_shortest_path = limit_shortest_path

        self.nx_graph: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "heuristic"

        self.solving_instance = SolvingInstance(self.problem_instance)

    def run(
        self,
    ):
        start_time = time.time()

        self.solving_instance.basic_placement_and_routing()
        self.solving_instance.render_solution()

        self.runtime = time.time() - start_time
        self.score = self.solving_instance.get_score()
        self.x_variable, self.y_variable, self.q_variable = (
            self.solving_instance.get_arc_variables()
        )

    def compute_first_shortest_path(self) -> list:
        print(self.problem_instance.origins)
        return [
            self._select_shortest_path(
                self.problem_instance.origins[demand_index],
                self.service_placement[
                    self.problem_instance.demand_to_service[demand_index]
                ],
                0,
            )
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]

    def update_variables(self):
        self.y_variable = np.zeros(
            (self.problem_instance.number_of_service, self.nx_graph.number_of_nodes()),
            dtype=int,
        )
        for i, col_index in enumerate(self.service_placement):
            self.y_variable[i, col_index] = 1

        self.x_variable = np.zeros(
            (
                self.problem_instance.total_number_of_demands,
                self.nx_graph.number_of_edges(),
            ),
            dtype=int,
        )
        for demand_index, path in enumerate(self.paths_used):
            for edge in path_node_to_path_tuple(path):
                edge_index = self.problem_instance.network_graph.arc_index_dictionary[
                    edge
                ]
                self.x_variable[demand_index, edge_index] = 1

        self.q_variable = np.array(self.choosen_quantities)

    def augmentation_possible(self) -> int:
        """
        Trouver la demande augmentable associée au plus court chemin.
        On itère sur les demandes dans l'ordre des plus courts chemins
        """
        self._test_solution()
        longueur_chemins = [len(path) for path in self.paths_used]
        demand_indices = sorted(
            range(self.problem_instance.total_number_of_demands),
            key=lambda index: longueur_chemins[index],
        )
        for demand_index in demand_indices:
            if (
                self.choosen_quantities[demand_index]
                >= self.problem_instance.max_flows[demand_index]
            ):
                continue
            path_used = self.paths_used[demand_index]
            if len(path_used) <= 1:
                return demand_index
            else:
                path_used = path_node_to_path_tuple(path_used)
                capacite_a_la_limite = np.all(
                    [
                        self.nx_graph.edges[edge]["remaining_capacity"] > 0
                        for edge in path_used
                    ]
                )
                latence_sur_le_chemin = sum(
                    [self.nx_graph.edges[edge]["latency"] for edge in path_used]
                )
                latence_beta = sum(
                    [self.nx_graph.edges[edge]["beta"] for edge in path_used]
                )
                latence_a_la_limite = (
                    latence_sur_le_chemin
                    <= self.problem_instance.latencies[demand_index] - latence_beta
                )

                if not capacite_a_la_limite and not latence_a_la_limite:
                    return demand_index
        return -1

    def find_problematic_demand_path(self):
        """
        On cherche le numéro de la demande qui pose problème.

        Pour trouver une demand problématique, on itère sur d.
        On check ensuite si les capacités sont ok
        et si la latence est ok.
        On assert False à la fin car on est sûr
        qu'il existe une demande problématique.
        """
        for demand_index in range(self.problem_instance.total_number_of_demands):
            demand_path = self.paths_used[demand_index]
            latency = sum(
                [
                    self.nx_graph.edges[edge]["latency"]
                    for edge in path_node_to_path_tuple(demand_path)
                ]
            )
            if latency > self.problem_instance.latencies[demand_index]:
                return demand_index
            for edge in path_node_to_path_tuple(demand_path):
                if self.nx_graph.edges[edge]["remaining_capacity"] < 0:
                    return demand_index

        return -1  # Not a wrong demand path, but a service node wrong association

    def _assert_all_constraints(self):
        # Contraintes :
        # - Capacité pour chaque arc
        # - Pour chaque demande, quantité et latence limite
        # - Ressource pour chaque noeud
        for edge in self.nx_graph.edges:
            capacity_condition = self.nx_graph.edges[edge]["remaining_capacity"] >= 0
            assert capacity_condition, "Capacité non respectée."

        for demand_index in range(self.problem_instance.total_number_of_demands):
            flows_limits_condition = (
                self.problem_instance.min_flows[demand_index]
                <= self.choosen_quantities[demand_index]
                <= self.problem_instance.max_flows[demand_index]
            )
            assert flows_limits_condition, "Max flow exceeded."

            latency_demand = sum(
                self.nx_graph.edges[edge]["latency"]
                for edge in path_node_to_path_tuple(self.paths_used[demand_index])
            )
            latency_condition = (
                latency_demand <= self.problem_instance.latencies[demand_index]
            )
            assert latency_condition, "Latency exceeded."

        for node in self.nx_graph.nodes:
            remaining_resources_condition = (
                self.nx_graph.nodes[node]["remaining_resources"] >= 0
            )
            assert remaining_resources_condition, "No enough resources."

    def _update_edge_attributes(self):
        # Sur chaque Edge : quantité, latence, remaining quantity
        for demand_index in range(self.problem_instance.total_number_of_demands):
            for edge in path_node_to_path_tuple(self.paths_used[demand_index]):
                # Update Q
                self.nx_graph.edges[edge]["quantity"] += self.choosen_quantities[
                    demand_index
                ]
                # Update latence
                self.nx_graph.edges[edge]["latency"] += (
                    self.choosen_quantities[demand_index]
                    * self.nx_graph.edges[edge]["alpha"]
                )

        # Update remaining quantities using capacity and quantity.
        for edge in self.nx_graph.edges:
            self.nx_graph.edges[edge]["remaining_capacity"] = (
                self.nx_graph.edges[edge]["capacity"]
                - self.nx_graph.edges[edge]["quantity"]
            )

    def _reset_edge_attributes(self):
        for edge in self.nx_graph.edges:
            # Reset Q, l
            self.nx_graph.edges[edge]["quantity"] = 0
            self.nx_graph.edges[edge]["latency"] = self.nx_graph.edges[edge]["beta"]

    def _update_node_attributes(self):
        for service_index in range(self.problem_instance.number_of_service):
            target_node = self.service_placement[service_index]
            required_resources = self.problem_instance.needed_resources[service_index]
            self.nx_graph.nodes[target_node][
                "remaining_resources"
            ] -= required_resources

    def _reset_node_attributes(self):
        for node in self.nx_graph.nodes:
            # Update remaining resources
            self.nx_graph.nodes[node]["remaining_resources"] = self.nx_graph.nodes[
                node
            ]["available_resources"]

    def _select_shortest_path(self, source: int, destination: int, index: int):
        index = int(index)
        shortest_paths = nx.algorithms.simple_paths.shortest_simple_paths(
            self.nx_graph,
            source,
            destination,
        )
        for i, path in enumerate(shortest_paths):
            if i == index:
                return path
            last_visited_path = path
        return last_visited_path

    def _plot_solution(self):
        """
        Idée:
        Plot le graphe avec ses attributs.
        Plot la solution actuelle. Pourrait surcharger le graph
        """
        pass

    def _services_nodes_association(self):
        needed_resources = sorted(
            [
                (service_index, res)
                for service_index, res in enumerate(
                    self.problem_instance.needed_resources
                )
            ],
            key=lambda elem: elem[1],
        )
        av_resources = [
            self.nx_graph.nodes[node]["available_resources"]
            for node in self.nx_graph.nodes
        ]
        for service_index, needed_res in needed_resources:
            selected_node = np.argmax(av_resources)
            av_resources[selected_node] -= needed_res
            assert (
                av_resources[selected_node] >= 0
            ), "Greedy node-service association failed."
            self.service_placement[service_index] = selected_node

    def _update_quantities(self):
        max_iter = 100
        for _ in range(max_iter):
            demand_index = rnd_gen.integers(
                self.problem_instance.total_number_of_demands
            )
            self.choosen_quantities[demand_index] += 1
            try:
                self._test_solution()
            except AssertionError:
                self.choosen_quantities[demand_index] -= 1
                self._reset_edge_attributes()
                self._reset_node_attributes()
                self._update_edge_attributes()
                self._update_node_attributes()

    def _test_solution(self):
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes()
        self._update_node_attributes()
        self._assert_all_constraints()
