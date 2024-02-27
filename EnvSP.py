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
    ):
        # Problem data
        self.problem_instance = problem_instance
        self.sp_number = sp_number
        self.limit_step = limit_step

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
        selected_service, choosen_node, selected_demand, action_q, action_sp = action

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
            if (
                action_sp == 1
            ):  # PCC pas changé, on peut changer la quantité d'une demande
                if action_q == 1:  # Q change pas
                    pass
                elif action_q == 0:  # On baisse Q
                    if self.choosen_quantities[selected_demand] > 1:
                        self.choosen_quantities[selected_demand] -= 1
                        self.latencies[
                            selected_demand
                        ] -= self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )

                    obs = self.state
                else:  # Si on ne depasse pas Q, et que la latence sur le chemin ne dépasse pas la limite, on peut augmenter Q
                    if (
                        self.choosen_quantities[selected_demand]
                        < self.problem_instance.max_flows[selected_demand]
                    ) and (
                        self.latencies[selected_demand]
                        + self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )
                        < self.problem_instance.latencies[selected_demand]
                    ):
                        self.choosen_quantities[selected_demand] += 1
                        self.latencies[
                            selected_demand
                        ] += self._compute_latency_on_path(
                            1, self.paths_used[selected_demand]
                        )
            elif action_sp == 0:  # Shortest path changée
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
        reward = self.choosen_quantities.sum()

        if reward > self.max_q_reached:
            self.max_q_reached = reward
            print(self.max_q_reached)

        reward -= self.max_q_reached
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
