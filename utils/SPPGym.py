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


class SPPGym(gym.Env):
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

        self._make_action_observation_spaces()

        self.solving_instance = SolvingInstance(problem_instance)

        self.solving_instance.basic_placement_and_routing()

        self.best_solution = self.solving_instance.get_arc_variables()
        self.max_q_reached = self.solving_instance.get_score()

    def _build_initial_solution(self) -> tuple:
        """
        Build the initial solution for initialization of the environment.
        """
        # State : Ou sont placés les services (taille S, max n)
        #       x Quel est le numéro du PCC pour chaque demande (taille d, max sp_number)
        #       x Quantité de chaque demande (taille d, max q_dmax)

        # On place tout sur le noeud le plus large
        node_with_max_resources = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )
        service_placement = [
            node_with_max_resources
        ] * self.problem_instance.number_of_service

        # The shortest path indice for each demand
        sp_indices = [0] * self.problem_instance.total_number_of_demands

        # Min flows
        choosen_quantities = self.problem_instance.min_flows

        # On check quand même la solution au cas où
        observation = service_placement + sp_indices + choosen_quantities
        self.problem_instance._set_arc_solution(
            *self.observation_to_variables(observation)
        )

        self.problem_instance.check_current_solution()

        return observation

    def observation_to_variables(self, observation: list) -> tuple:
        """
        Observation to (x, y, q).
        """
        # State : Ou sont placés les services (taille S, max n)
        #       x Quel est le numéro du PCC pour chaque demande (taille d, max sp_number)
        #       x Quantité de chaque demande (taille d, max q_dmax)
        service_placement, sp_indices, choosen_quantities = (
            self.observation_to_components(observation)
        )

        x_var = np.zeros(
            (
                self.problem_instance.total_number_of_demands,
                self.graph.number_of_edges(),
            )
        )
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            demand_path = path_node_to_path_tuple(
                self._select_shortest_path(origin, target, sp_indices[demand_index])
            )
            for edge in demand_path:
                edge_index = self.network_graph.arc_index_dictionary[edge]
                x_var[demand_index, edge_index] = 1.0
        y_var = np.zeros(
            (self.problem_instance.number_of_service, self.graph.number_of_nodes())
        )
        for service_index in range(self.problem_instance.number_of_service):
            y_var[service_index, service_placement[service_index]] = 1.0
        q_var = np.array(choosen_quantities)
        return (x_var, y_var, q_var)

    def observation_to_components(self, observation: list) -> tuple:
        """
        Observation to (service_placement, sp_indices, quantities)
        """
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands
        service_placement = observation[:S]
        sp_indices = observation[S : S + D]
        choosen_quantities = observation[S + D :]
        return service_placement, sp_indices, choosen_quantities

    def reset(self, seed=0):
        # Placement aléatoire, PCC0 et quantités 1
        self.steps_done = 0
        self.solving_instance.reset_instance()

        obs = self.solving_instance.get_actual_state()
        info = {}

        self.recent_scores = [-10 for _ in range(self.limit_step)]
        self.max_q_reached_episode = 0

        return obs, info

    def _update_placement(self, selected_service: int, choosen_node: int) -> None:
        """
        Change the placement of service.
        If the same SP indices do not work,
        take back the previous placement
        """
        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )
        old_choosen_node = service_placement[selected_service]
        service_placement[selected_service] = choosen_node

        # On doit update les shortest paths
        # parce que les chemins actuels ne fonctionnent plus necessaire
        # Mais bon, on garde les mêmes indices, et si jamais le changement de service fonctionne pas
        # Ba on déplace pas le service
        # On vérifie donc que le changement est bien possible
        self.state = service_placement + sp_indices + quantities

        try:
            self._test_solution()
        except AssertionError:
            service_placement[selected_service] = old_choosen_node
            self.state = service_placement + sp_indices + quantities

    def _reroute(self, selected_demand: int, choosen_sp: int):
        """
        choosen_sp is the index of the shortest path choosent through this action.
        """
        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )
        old_sp_index = sp_indices[selected_demand]
        sp_indices[selected_demand] = choosen_sp

        self.state = service_placement + sp_indices + quantities

        try:
            self._test_solution()
        except AssertionError:
            sp_indices[selected_demand] = old_sp_index
            self.state = service_placement + sp_indices + quantities

    def _update_quantity(self, selected_demand: int, action_q: int):
        """
        For a given demand, apply +-1 or 0 along action_q.
        """
        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )
        old_quantity = quantities[selected_demand]
        min_flow = self.problem_instance.min_flows[selected_demand]
        max_flow = self.problem_instance.max_flows[selected_demand]

        if action_q == 0:  # On baisse Q
            quantities[selected_demand] = max(min_flow, quantities[selected_demand] - 1)
        elif action_q == 2:
            quantities[selected_demand] = min(
                max_flow,
                quantities[selected_demand] + 1,
            )

        self.state = service_placement + sp_indices + quantities
        try:
            self._test_solution()
        except AssertionError:
            quantities[selected_demand] = old_quantity
            self.state = service_placement + sp_indices + quantities

    def render(self):
        self.solving_instance.render_solution()

    def step(self, action: tuple):
        self.steps_done += 1
        done = False
        terminated = self.steps_done >= self.limit_step
        old_state = self.solving_instance.get_actual_state()
        info = {}

        # Action : Choix_Action x Choix_Service x Placement x Choix_Demande x Changement_Quantité x Changement PCC
        (
            service_or_demand_or_quantities,
            selected_service,
            choosen_node_for_this_service,
            selected_demand,
            action_q,
            choosen_sp,
        ) = action

        # Si service_or_demand == 0, on s'occupe du service et on le déplace
        if service_or_demand_or_quantities == 0:
            new_demands_path = [1 for _ in range(self.problem_instance.demand_per_service)]
            self.solving_instance.replace_service(selected_service, choosen_node_for_this_service, new_demands_path)
            reward = -10 if old_state == self.state else -2
        elif service_or_demand_or_quantities == 1:  # Sinon, on reroute la demande
            old_state = self.state
            self._reroute(selected_demand, choosen_sp)
            reward = -10 if old_state == self.state else -1
        else:
            self._update_quantity(selected_demand, action_q)
            if self.max_q_reached_episode == 0:
                reward = max(
                    np.sum(self.state[-self.problem_instance.total_number_of_demands :])
                    - self.problem_instance.total_number_of_demands,
                    0,
                )
            else:
                reward = max(
                    np.sum(self.state[-self.problem_instance.total_number_of_demands :])
                    - self.max_q_reached_episode,
                    0,
                )

        obs = np.array(self.state)
        info = {}
        self._test_solution()

        objective = np.sum(self.state[-self.problem_instance.total_number_of_demands :])

        if objective > self.max_q_reached:
            print(objective)
            self.max_q_reached = objective
            self.best_solution = self.observation_to_variables(self.state)
            # print(self.max_q_reached)

        if objective > self.max_q_reached_episode:
            self.max_q_reached_episode = objective

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

    def _update_edge_attributes(self):
        """
        Update the edge attributes quantity, latency and remaining capacity along self.state.
        """
        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )

        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]

            origin = self.problem_instance.origins[demand_index]
            target = np.argmax(service_placement[service_index])
            demand_path = path_node_to_path_tuple(
                self._select_shortest_path(origin, target, sp_indices[demand_index])
            )
            for edge in demand_path:
                # Update quantity
                self.graph.edges[edge]["quantity"] += quantities[demand_index]
                # Update latency
                self.graph.edges[edge]["latency"] += (
                    quantities[demand_index] * self.graph.edges[edge]["alpha"]
                )  # Le beta y est deja

        for edge in self.graph.edges:
            # Update remaining capacity
            self.graph.edges[edge]["remaining_capacity"] = (
                self.graph.edges[edge]["capacity"] - self.graph.edges[edge]["quantity"]
            )

    def _reset_edge_attributes(self):
        """
        Set edge["quantity"] to 0 and edge["latency"] to edge["beta"], the base latency.
        """
        for edge in self.graph.edges:
            # Reset Q, l
            self.graph.edges[edge]["quantity"] = 0
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]

    def _update_node_attributes(self):
        service_placement, _, _ = self.observation_to_components(self.state)

        for service_index in range(self.problem_instance.number_of_service):
            target_node = service_placement[service_index]
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

        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )

        for edge in self.graph.edges:
            assert (
                self.graph.edges[edge]["remaining_capacity"] >= 0
            ), "Capacité non respectée."

        for demand_index in range(self.problem_instance.total_number_of_demands):
            assert (
                quantities[demand_index]
                <= self.problem_instance.max_flows[demand_index]
            ), "Max flow exceeded."
            latency = 0
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            demand_path = path_node_to_path_tuple(
                self._select_shortest_path(origin, target, sp_indices[demand_index])
            )
            for edge in demand_path:
                latency += self.graph.edges[edge]["latency"]
            assert (
                latency <= self.problem_instance.latencies[demand_index]
            ), "Latency exceeded."

        for node in self.graph.nodes:
            assert (
                self.graph.nodes[node]["remaining_resources"] >= 0
            ), "No enough resources."

    def _test_solution(self):
        """
        Test if a given solution self.state is doable or not.
        - Capacity conditions
        - Latency conditions
        - Max flow conditions
        - Resources conditions
        """
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes()
        self._update_node_attributes()
        self._assert_all_constraints()

    def _make_action_observation_spaces(self):
        """
        Build the Gym action and observation spaces.
        """
        # Action :
        #   Choix action (service demande ou quantité)
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
        low_array = np.array(
            [0]
            * (
                self.problem_instance.number_of_service
                + 2 * self.problem_instance.total_number_of_demands
            )
        )
        high_array = np.array(
            [self.graph.number_of_nodes()] * self.problem_instance.number_of_service
            + [self.sp_number] * self.problem_instance.total_number_of_demands
            + self.problem_instance.max_flows
        )
        self.observation_space = Box(low=low_array, high=high_array)

    def clip_state(self):
        self.state = np.round(self.state).astype(int)
        service_placement, sp_indices, quantities = self.observation_to_components(
            self.state
        )
        service_placement = service_placement.clip(0, self.graph.number_of_nodes() - 1)
        sp_indices = sp_indices.clip(0, self.sp_number - 1)
        quantities = quantities.clip(0, self.problem_instance.max_flows[0])
        self.state = np.concatenate([service_placement, sp_indices, quantities])
