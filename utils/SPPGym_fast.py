import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Box, MultiDiscrete
import networkx as nx
from typing import List
import itertools

from problem_classes.probleminstance import ProblemInstance
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node
import matplotlib.pyplot as plt
from problem_classes.solvinginstance import SolvingInstance


class SPPGymFast(gym.Env):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        sp_number: int = 10,  # Max number of explored shortest paths
        limit_step: int = 2000,  # Limit of the number of evolving steps
    ):
        # Problem data
        self.problem_instance = problem_instance
        self.sp_number = sp_number
        self.limit_step = limit_step

        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph

        self.solving_instance = SolvingInstance(problem_instance)
        self.solving_instance.basic_placement_and_routing()
        self.best_solution = self.solving_instance.get_arc_variables()
        self.max_q_reached = self.solving_instance.get_score()

        self._make_action_observation_spaces()

    def reset(self, seed=0):
        # Placement aléatoire, PCC0 et quantités 1
        self.steps_done = 0
        self.solving_instance.reset_instance()
        self.solving_instance.basic_placement_and_routing()
        obs = self.solving_instance.get_actual_state()
        self.state = obs

        info = {}

        self.recent_scores = [-10 for _ in range(self.limit_step)]
        self.max_q_reached_episode = 0

        return obs, info

    def render(self):
        self.solving_instance.render_solution()

    def step(self, action: tuple):
        self.steps_done += 1
        done = False
        terminated = self.steps_done >= self.limit_step
        info = {}

        (
            service_or_demand_or_quantities,  # Choix_Action
            selected_service,  # Choix_Service
            choosen_node_for_this_service,  # Placement
            selected_demand,  # Choix_Demande
            action_q,  # Changement_Quantité
            choosen_sp,  # Changement PCC
        ) = action

        # Si service_or_demand == 0, on s'occupe du service et on le déplace
        if service_or_demand_or_quantities == 0:

            demands_in_service = (
                self.problem_instance.demand_per_service[selected_service] * selected_service,
                self.problem_instance.demand_per_service[selected_service] * (selected_service + 1),
            )
            paths = []
            for demand_index in range(*demands_in_service):
                origin = self.problem_instance.origins[demand_index]
                path = nx.shortest_path(
                    self.graph,
                    origin,
                    choosen_node_for_this_service,
                ) if origin != choosen_node_for_this_service else []
                paths.append(path)

            try:
                self.solving_instance.replace_service(selected_service, choosen_node_for_this_service, paths)
            except AssertionError:
                reward = -1

        elif service_or_demand_or_quantities == 1:  # Sinon, on reroute la demande
            self.solving_instance._all_demands_are_routed()
            self.solving_instance.reroute_demand(
                selected_demand,
                self._select_shortest_path(
                    self.problem_instance.origins[selected_demand],
                    self.solving_instance.service_placement[self.problem_instance.demand_to_service[selected_demand]],
                    choosen_sp,
                ),
            )
        else: # Troisième possibilité : on mets à jour les quantités avec + ou - 1
            if action_q == 0:
                self.solving_instance.decrease_quantity(selected_demand)
            elif action_q == 2:
                self.solving_instance.increase_quantity(selected_demand)

        obs = np.array(self.state)
        info = {}

        objective = np.sum(self.state[-self.problem_instance.total_number_of_demands :])

        if objective > self.max_q_reached:
            print(objective)
            self.max_q_reached = objective
            self.best_solution = self.observation_to_variables(self.state)
            # print(self.max_q_reached)

        if objective > self.max_q_reached_episode:
            self.max_q_reached_episode = objective

        reward = self.solving_instance.get_score()

        self.recent_scores[self.steps_done - 1] = reward

        if terminated:
            if hasattr(self, "episode"):
                self.episode += 1
            else:
                self.episode = 0

            if self.episode % 5 == 0:
                plt.cla()

            rolling = self.limit_step // 50
            self.recent_scores = np.convolve(
                self.recent_scores, np.ones(rolling) / rolling, mode="valid"
            )
            plt.plot(self.recent_scores, label="Episode {}".format(self.episode))
            plt.legend()
            plt.savefig("results.png")

        return obs, reward / 100, done, terminated, info

    def _select_shortest_path(self, source: int, destination: int, index: int):
        """
        Returns the index-th shortest path from source to destination.
        """
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

    def _make_action_observation_spaces(self):
        """
        Build the Gym action and observation spaces.
        """
        # Action :
        #   Choix action (service,  demande ou quantité)
        # x Choix_Service
        # x Placement
        # x Choix_Demande
        # x Changement_Quantité
        # x Choix PCC
        self.action_space = MultiDiscrete(
            [
                3,
                self.problem_instance.number_of_service,
                self.graph.number_of_nodes(),
                self.problem_instance.total_number_of_demands,
                3,
                self.sp_number,
            ]
        )
        # State : Ou sont placés les services (taille S, max n)
        #       x Quel est le numéro du PCC pour chaque demande (taille d, max sp_number)
        #       x Quantité de chaque demande (taille d, max q_dmax)

        low_array = np.array([0.0] * self.solving_instance.get_state_dimension())
        high_quantity = np.inf
        high_array = np.array(
            [high_quantity] * self.solving_instance.get_state_dimension()
        )
        self.observation_space = Box(low=low_array, high=high_array)
