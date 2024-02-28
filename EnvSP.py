import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Box, MultiDiscrete
from problem_classes.probleminstance import ProblemInstance
import networkx as nx
from typing import List
import itertools


class EnvSP(gym.Env):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        sp_number: int = 3,
        limit_step: int = 10000,
        verbose: bool = True,
    ):
        # Problem data
        self.problem_instance = problem_instance
        self.sp_number = sp_number
        self.limit_step = limit_step
        self.verbose = verbose

        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph

        # Action : Choix_Service x Placement x Choix_Demande x Changement_Quantité x Changement PCC
        # self.action_space = Tuple(
        #     (
        #         Discrete(self.problem_instance.number_of_service),
        #         Discrete(self.graph.number_of_nodes()),
        #         Discrete(self.problem_instance.total_number_of_demands),
        #         Discrete(3),
        #         Discrete(3),
        #     )
        # )
        self.action_space = MultiDiscrete(
            [
                max(
                    3,
                    self.problem_instance.number_of_service,
                    self.graph.number_of_nodes(),
                    self.problem_instance.total_number_of_demands,
                )
                for _ in range(5)
            ]
        )

        # State : Ou sont placés les services x Quel est le numéro du PCC pour chaque demande x Quantité de chaque demande
        # Vecteur de taille : Serv + Dem + Dem
        self.observation_space = Box(
            low=0.0,
            high=np.inf,
            shape=(
                self.problem_instance.number_of_service
                + 2 * self.problem_instance.total_number_of_demands,
            ),
        )

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
            nx.shortest_path(
                self.graph,
                self.problem_instance.origins[demand_index],
                self.service_placement[
                    self.problem_instance.demand_to_service[demand_index]
                ],
            )
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]
        self.latencies = [
            self._compute_latency_on_path(
                self.choosen_quantities[demand_index], self.paths_used[demand_index]
            )
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]
        self.quantity_by_edge = {edge: 0.0 for edge in self.graph.edges}
        self._update_quantity_by_edge()

        self._compute_remaining_capacities()

    def reset(self, seed=0):
        # Placement aléatoire, PCC0 et quantités 1
        self.state = np.concatenate(
            (self.service_placement, self.sp_indices, self.choosen_quantities)
        )
        self.steps_done = 0
        self.max_q_reached = 0

        obs = self.state
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        selected_service, choosen_node, selected_demand, action_q, action_sp = action

        if (
            selected_service >= self.problem_instance.number_of_service
            or choosen_node >= self.graph.number_of_nodes()
            or selected_demand >= self.problem_instance.total_number_of_demands
            or action_q >= 3
            or action_sp >= 3
        ):
            reward -= 10

        # if np.random.random() > 1e-3:
        #     selected_service = 0
        #     choosen_node = self.service_placement[0]
        #     action_sp = 1

        selected_service = int(
            np.clip(selected_service, 0, self.problem_instance.number_of_service - 1)
        )
        choosen_node = int(np.clip(choosen_node, 0, self.graph.number_of_nodes() - 1))
        selected_demand = int(
            np.clip(
                selected_demand, 0, self.problem_instance.total_number_of_demands - 1
            )
        )
        action_q = int(np.clip(action_q, 0, 2))
        action_sp = int(np.clip(action_sp, 0, 2))

        done = False
        self.steps_done += 1
        terminated = self.steps_done >= self.limit_step
        info = {}

        # Contraintes à respecter : latence de chaque demande, quantité maximum de chaque demande

        # Question : est-ce qu'on change le placement de ce service ?
        if self.service_placement[selected_service] == choosen_node:
            # Placement pas changé. On peut changer le PCC
            if self.verbose:
                print("Placement du service {} non changée.".format(selected_service))

            if (
                action_sp == 1
            ):  # PCC pas changé, on peut changer la quantité d'une demande

                if self.verbose:
                    print("PCC de la demande {} non changé.".format(selected_demand))

                if action_q == 1:  # Q change pas
                    if self.verbose:
                        print("Quantité inchangée.")
                elif action_q == 0:  # On baisse Q
                    if self.verbose:
                        print("Quantité diminuée.")
                    if self.choosen_quantities[selected_demand] > 1:
                        self.choosen_quantities[selected_demand] -= 1
                        self.latencies[
                            selected_demand
                        ] -= self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )

                else:  # Si on ne depasse pas Q, et que la latence sur le chemin ne dépasse pas la limite et qu'on a une capacité restant sur chaque arc, on peut augmenter Q

                    capacity_condition = self._update_remaining_capacities_demand(
                        1, self.paths_used[selected_demand]
                    )
                    max_flow_condition = (
                        self.choosen_quantities[selected_demand]
                        < self.problem_instance.max_flows[selected_demand]
                    )
                    latency_condition = (
                        self.latencies[selected_demand]
                        + self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )
                        < self.problem_instance.latencies[selected_demand]
                    )

                    print(capacity_condition, max_flow_condition, latency_condition)

                    if max_flow_condition and latency_condition and capacity_condition:
                        if self.verbose:
                            print("Quantité augmentée de 1.")
                        self.choosen_quantities[selected_demand] += 1
                        self.latencies[
                            selected_demand
                        ] += self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )
            elif action_sp == 0:  # Shortest path changée
                reward -= 1
                if self.sp_indices[selected_demand] > 1:
                    self.sp_indices[selected_demand] -= 1
                    self.choosen_quantities[selected_demand] = 1
                    self.paths_used[selected_demand] = self._select_shortest_path(
                        self.problem_instance.origins[selected_demand],
                        self.service_placement[
                            self.problem_instance.demand_to_service[selected_demand]
                        ],
                        self.sp_indices[selected_demand],
                    )
                    self.latencies[selected_demand] = self._compute_latency_on_path(
                        1, self.paths_used[selected_demand]
                    )
            else:
                reward -= 1
                if self.sp_indices[selected_demand] < self.sp_number:
                    self.sp_indices[selected_demand] += 1
                    self.choosen_quantities[selected_demand] = 1
                    self.paths_used[selected_demand] = self._select_shortest_path(
                        self.problem_instance.origins[selected_demand],
                        self.service_placement[
                            self.problem_instance.demand_to_service[selected_demand]
                        ],
                        self.sp_indices[selected_demand],
                    )
                    self.latencies[selected_demand] = self._compute_latency_on_path(
                        1, self.paths_used[selected_demand]
                    )
        else:
            reward -= 10

            # Placement changé
            self.service_placement[selected_service] = choosen_node
            # Ideal : pour chaque demande du service, mettre des indices 1
            for demand_index in range(self.problem_instance.total_number_of_demands):
                if (
                    self.problem_instance.demand_to_service[demand_index]
                    == selected_service
                ):
                    self.sp_indices[demand_index] = 0
                    self.choosen_quantities[demand_index] = 1
                    self.paths_used[demand_index] = nx.shortest_path(
                        self.graph,
                        self.problem_instance.origins[demand_index],
                        choosen_node,
                    )
                    self.latencies[demand_index] = self._compute_latency_on_path(
                        1, self.paths_used[demand_index]
                    )

        self.state = np.concatenate(
            (
                self.service_placement,
                self.sp_indices,
                self.choosen_quantities,
            )
        )

        obs = self.state
        reward += self.choosen_quantities.sum()

        if self.choosen_quantities.sum() > self.max_q_reached:
            self.max_q_reached = self.choosen_quantities.sum()
            print(self.max_q_reached)

        reward /= 100

        return obs, reward, done, terminated, info

    def render(self):
        print()
        print("Number of services : {}".format(self.problem_instance.number_of_service))
        print(
            "Number of demands : {}".format(
                self.problem_instance.total_number_of_demands
            )
        )
        print()
        print("Service placement : ")
        print(self.service_placement)
        print()
        print("Shortest paths indices :")
        print(self.sp_indices)
        print()
        print("Choosen quantities :")
        print(self.choosen_quantities)
        print()
        # print("Path used :")
        # for path in self.paths_used:
        #     print(path)
        # print()
        print("Latencies :")
        print(np.round(self.latencies, 2))
        print()

    def _compute_latency_on_path(self, quantity: int, path: List[int]):
        return sum(
            self.graph.edges[(path[path_index], path[path_index + 1])]["alpha"]
            * quantity
            + self.graph.edges[(path[path_index], path[path_index + 1])]["beta"]
            for path_index in range(len(path) - 1)
        )

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

    def _reset_remaining_capacities(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]

    def _update_remaining_capacities_demand(
        self, quantity: int, path: List[int]
    ) -> bool:
        for index in range(len(path) - 1):
            edge = (path[index], path[index + 1])
            self.graph.edges[edge]["remaining_capacity"] -= quantity
            if self.graph.edges[edge]["remaining_capacity"] < 0:
                return False
        return True

    def _compute_remaining_capacities(self):
        self._reset_remaining_capacities()
        for demand_index in range(self.problem_instance.total_number_of_demands):
            path = self.paths_used[demand_index]
            quantity = self.choosen_quantities[demand_index]
            updated = self._update_remaining_capacities_demand(quantity, path)
            assert updated, "The capacity condition is not ensured."

    def _update_edges_attributes(self):
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

    def reset_edge_attributes(self):
        for edge in self.graph.edges:
            # Reset Q, l
            self.graph.edges[edge]["quantity"] = 0
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]


from utils.data_management import (
    get_instance_list,
)

instance_list = get_instance_list()
instance = instance_list[0]

env = EnvSP(instance)
env.reset()

action = (0, env.service_placement[0], 0, 2, 1)

for _ in range(10):
    env.step(action)
