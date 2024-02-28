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
        limit_step: int = 1000,
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
        self.steps_done += 1
        selected_service, choosen_node, selected_demand, action_q, action_sp = action

        if (
            selected_service >= self.problem_instance.number_of_service
            or choosen_node >= self.graph.number_of_nodes()
            or selected_demand >= self.problem_instance.total_number_of_demands
            or action_q >= 3
            or action_sp >= 3
        ):
            obs = self.state
            reward = -1
            done = False
            terminated = self.steps_done >= self.limit_step
            info = {}
            return obs, reward, done, terminated, info

        elif self.service_placement[selected_service] != choosen_node:
            reward = -0.2
        elif action_sp != -1:
            reward = -0.1
        else:
            reward = 0

        if np.random.random() > 1e-3:
            selected_service = 0
            choosen_node = self.service_placement[0]
            action_sp = 1

        action = self._clip_action(action)
        selected_service, choosen_node, selected_demand, action_q, action_sp = action

        if (
            self.service_placement[selected_service] == choosen_node
        ):  # Placement inchangé
            if action_sp == 0:  # Rerouting -1
                self.sp_indices[selected_demand] = max(
                    0, self.sp_indices[selected_demand] - 1
                )
                old_path = self.paths_used[demand_index]

                origin = self.problem_instance.origins[selected_demand]
                target = self.service_placement[
                    self.problem_instance.demand_to_service[selected_demand]
                ]
                index = self.sp_indices[selected_demand]
                self.paths_used[selected_demand] = self._select_shortest_path(
                    origin, target, index
                )
                try:
                    self._test_solution()
                except AssertionError:
                    self.paths_used[demand_index] = old_path
                    self.sp_indices[selected_demand] += 1

            elif action_sp == 1:  # Pas de rerouting
                if action_q == 1:  # Q change pas
                    pass
                elif action_q == 0:  # On baisse Q
                    self.choosen_quantities[selected_demand] = max(
                        1, self.choosen_quantities[selected_demand] - 1
                    )
                    self._test_solution()  # En baissant les quantités, normalement rien n'arrive.

                else:  # On augmente Q
                    self.choosen_quantities[selected_demand] = min(
                        self.problem_instance.max_flows[selected_demand],
                        self.choosen_quantities[selected_demand] + 1,
                    )
                    try:
                        self._test_solution()
                    except AssertionError:
                        self.choosen_quantities[selected_demand] -= 1
            else:  # Rerouting +1
                self.sp_indices[selected_demand] = max(
                    0, self.sp_indices[selected_demand] + 1
                )
                old_path = self.paths_used[demand_index]

                origin = self.problem_instance.origins[selected_demand]
                target = self.service_placement[
                    self.problem_instance.demand_to_service[selected_demand]
                ]
                index = self.sp_indices[selected_demand]
                self.paths_used[selected_demand] = self._select_shortest_path(
                    origin, target, index
                )
                try:
                    self._test_solution()
                except AssertionError:
                    self.paths_used[demand_index] = old_path
                    self.sp_indices[selected_demand] -= 1

        else:  # Placement changé
            old_choosen_node = self.service_placement[selected_service]

            self.service_placement[selected_service] = choosen_node
            # Ideal : pour chaque demande du service, mettre des indices 0 au sp
            for demand_index in range(self.problem_instance.total_number_of_demands):
                if (
                    self.problem_instance.demand_to_service[demand_index]
                    == selected_service
                ):
                    old_sp_index = self.sp_indices[demand_index]
                    old_choosen_quantities = self.choosen_quantities[demand_index]
                    old_path_used = self.paths_used[demand_index]

                    self.sp_indices[demand_index] = 0
                    self.choosen_quantities[demand_index] = 1
                    origin = self.problem_instance.origins[demand_index]
                    target = choosen_node
                    index = 0
                    self.paths_used[demand_index] = self._select_shortest_path(
                        origin, target, index
                    )

                    try:
                        self._test_solution()
                    except AssertionError:
                        self.service_placement[selected_service] = old_choosen_node

                        self.sp_indices[demand_index] = old_sp_index
                        self.choosen_quantities[demand_index] = old_choosen_quantities
                        self.paths_used[demand_index] = old_path_used
                        continue

        self._test_solution()

        self.state = np.concatenate(
            (
                self.service_placement,
                self.sp_indices,
                self.choosen_quantities,
            )
        )

        obs = self.state
        reward += self.choosen_quantities.sum() / 100

        if self.choosen_quantities.sum() > self.max_q_reached:
            self.max_q_reached = self.choosen_quantities.sum()
            print(self.max_q_reached)

        done = False
        terminated = self.steps_done >= self.limit_step
        info = {}
        return obs, reward, done, terminated, info

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

    def _test_solution(self):
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes()
        self._update_node_attributes()
        self._assert_all_constraints()

    def _clip_action(self, action: tuple) -> tuple:
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

        action = selected_service, choosen_node, selected_demand, action_q, action_sp
        return action


from utils.data_management import (
    get_instance_list,
)

instance_list = get_instance_list()
instance = instance_list[1]

env = EnvSP(instance)
env.reset()


for _ in range(1000):
    action = env.action_space.sample()
    env.step(action)
