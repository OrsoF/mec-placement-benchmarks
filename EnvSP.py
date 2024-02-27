import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Box
from problem_classes.probleminstance import ProblemInstance
import networkx as nx
from typing import List
import itertools


class EnvSP(gym.Env):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        sp_number: int = 3,
        limit_step: int = 100,
    ):
        # Problem data
        self.problem_instance = problem_instance
        self.sp_number = sp_number
        self.limit_step = limit_step

        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph

        # Action : Choix_Service x Placement x Choix_Demande x Changement_Quantité x Changement PCC
        self.action_space = Tuple(
            (
                Discrete(self.problem_instance.number_of_service),
                Discrete(self.graph.number_of_nodes()),
                Discrete(self.problem_instance.total_number_of_demands),
                Discrete(3),
                Discrete(3),
            )
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
        obs = self.state
        info = {}
        return obs, info

    def step(self, action):
        selected_service, choosen_node, selected_demand, action_q, action_sp = action
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
                    self.paths_used[selected_demand] = next(
                        itertools.islice(
                            nx.all_shortest_paths(
                                self.graph,
                                self.problem_instance.origins[selected_demand],
                                self.service_placement[
                                    self.problem_instance.demand_to_service[
                                        selected_demand
                                    ]
                                ],
                            ),
                            self.sp_indices[selected_demand],
                            None,
                        )
                    )
                    self.latencies = self._compute_latency_on_path(
                        1, self.paths_used[selected_demand]
                    )
            else:
                if self.sp_indices[selected_demand] < self.sp_number:
                    self.sp_indices[selected_demand] += 1
                    self.choosen_quantities[selected_demand] = 1
                    self.paths_used[selected_demand] = next(
                        itertools.islice(
                            nx.all_shortest_paths(
                                self.graph,
                                self.problem_instance.origins[selected_demand],
                                self.service_placement[
                                    self.problem_instance.demand_to_service[
                                        selected_demand
                                    ]
                                ],
                            ),
                            self.sp_indices[selected_demand],
                            None,
                        )
                    )
                    self.latencies = self._compute_latency_on_path(
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
                    self.sp_indices[demand_index] = 1
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

        if (
            not 0 <= selected_service < self.problem_instance.number_of_service
            or not 0 <= choosen_node < self.graph.number_of_nodes()
            or not 0 <= selected_demand < self.problem_instance.total_number_of_demands
            or not 0 <= action_q < 3
            or not 0 <= action_sp < 3
        ):
            reward = -100

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
        print("Path used :")
        for path in self.paths_used:
            print(path)
        print()
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


# from utils.data_management import (
#     get_instance_list,
# )

# instance_list = get_instance_list()
# instance = instance_list[0]
# env = EnvSP(instance)
# env.reset()

# env.render()

# for _ in range(1):
#     action = env.action_space.sample()
#     env.step(action)
#     env.render()
