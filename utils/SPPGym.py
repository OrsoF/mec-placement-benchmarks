import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, Discrete, Box, MultiDiscrete
import networkx as nx
from typing import List
import itertools
from problem_classes.solvinginstance import SolvingInstance
from problem_classes.probleminstance import ProblemInstance
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node
import matplotlib.pyplot as plt


moving_avg = lambda data, window: [
    sum(data[i : i + window]) / window for i in range(len(data) - window + 1)
]


class SPPGym(gym.Env):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        max_step: int,  # Limit of the number of evolving steps
        max_pcc: int = 3,  # Max number of explored shortest paths
        list_of_instances: list = None,  # Eventual other instances
    ):
        # Problem data
        self.problem_instance = problem_instance
        self.max_pcc = max_pcc
        self.max_step = max_step
        self.saved_shortest_path = {}
        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph
        self.list_of_instances = list_of_instances
        if not hasattr(self, "first_time"):
            self.initial_instance = problem_instance
        self.first_time = False

        # Gym
        self._make_action_observation_spaces()
        self.state = self._initial_state()

        # Evolution
        self.steps_done = 0
        self.rewards = [-10]
        self.scores = [0]
        self.solutions = [self._state_to_variables(self.state)]

    def _initial_state(self) -> list:
        """
        Build the initial state : basic placement, shortest path, smallest quantities.
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
        pcc_indices = [0] * self.problem_instance.total_number_of_demands

        # Min flows
        choosen_quantities = self.problem_instance.min_flows
        state = service_placement + pcc_indices + choosen_quantities
        return state

    def reset(self, seed: float = 0, initial_instance: bool = False):
        """Reset environment."""
        if self.list_of_instances is not None:
            new_instance = np.random.choice(self.list_of_instances)
            self.__init__(
                new_instance,
                self.max_step,
                self.max_pcc,
                self.list_of_instances,
            )

        if initial_instance:
            self.__init__(
                self.initial_instance,
                self.max_step,
                self.max_pcc,
                self.list_of_instances,
            )
        self.state = self._initial_state()
        self.steps_done = 0
        return self._observe(self.state), {}

    def _update_placement(
        self, state: list, selected_service: int, choosen_node: int
    ) -> Tuple:
        """
        Change the placement of service.
        If the same SP indices do not work,
        take back the previous placement
        """
        new_state = state.copy()
        new_state[selected_service] = choosen_node
        done = self._test_solution(new_state)
        if done:
            return new_state, done
        else:
            return state, done

    def _reroute(self, state: list, selected_demand: int, choosen_sp: int) -> Tuple:
        """
        choosen_sp is the index of the shortest path choosent through this action.
        """
        new_state = state.copy()
        new_state[self.problem_instance.number_of_service + selected_demand] = (
            choosen_sp
        )

        done = self._test_solution(new_state)
        if done:
            return new_state, done
        else:
            return state, done

    def _update_quantity(
        self, state: list, selected_demand: int, action_q: int
    ) -> tuple:
        """
        For a given demand, apply +-1 or 0 along action_q.
        """
        action_q = int(action_q == 2) - int(action_q == 0)
        new_state = state.copy()
        new_state[
            -self.problem_instance.total_number_of_demands + selected_demand
        ] += action_q

        done = self._test_solution(new_state)
        if done:
            return new_state, done
        else:
            return state, done

    def render(self):
        print(self.state)

    def step(self, action: tuple):
        self.steps_done += 1
        done = False
        terminated = self.steps_done >= self.max_step
        state = self.state

        # Action : Choix_Action x Choix_Service x Placement x Choix_Demande x Changement_Quantité x Changement PCC
        (
            service_or_demand_or_quantities,
            selected_service,
            choosen_node_for_this_service,
            selected_demand,
            action_q,
            choosen_sp,
        ) = action

        if (
            service_or_demand_or_quantities == 0
        ):  # On s'occupe du service et on le déplace
            state, done = self._update_placement(
                state, selected_service, choosen_node_for_this_service
            )
            reward = -1 if done else -10

        elif service_or_demand_or_quantities == 1:  # On reroute la demande
            state, done = self._reroute(state, selected_demand, choosen_sp)
            reward = -1 if done else -2

        else:  # On change Q
            state, done = self._update_quantity(state, selected_demand, action_q)
            # Reward:
            current_q = sum(self._state_to_vectors(state)[2])
            best_q = self._max_score_episode()
            reward = max(0, current_q - best_q)

        self.state = state
        obs = self._observe(state)
        info = {}

        self.rewards.append(reward)
        self.scores.append(sum(self._state_to_vectors(self.state)[2]))
        self.solutions.append(self._state_to_variables(self.state))

        if terminated:
            self.plot_average_reward()

        return obs, reward / 100, terminated, terminated, {}

    def plot_average_reward(self):
        self.count = 0 if not hasattr(self, "count") else self.count + 1
        episodes_count = len(self.rewards) // self.max_step
        rewards = self.rewards[: episodes_count * self.max_step]
        plt.cla()

        rewards_array = np.array(rewards).reshape((episodes_count, self.max_step))
        for index, rewards_list in enumerate(rewards_array[-3:]):
            rewards_list = moving_avg(rewards_list, 5)
            plt.plot(rewards_list, label=f"Episode {index + self.count} rewards")
            plt.plot(np.array(self.scores[-self.max_step :]) / 100)
        plt.legend()
        plt.savefig("results.png")

    def _select_shortest_path(self, source: int, destination: int, index: int) -> list:
        """
        Returns the index-th shortest path from source to destination.
        """
        index = int(index)

        if (source, destination, index) in self.saved_shortest_path:
            return self.saved_shortest_path[(source, destination, index)]

        for i, path in enumerate(
            nx.algorithms.simple_paths.shortest_simple_paths(
                self.graph,
                source,
                destination,
            )
        ):
            last_visited_path = path
            if i == index:
                break

        self.saved_shortest_path[(source, destination, index)] = last_visited_path
        return last_visited_path

    def _reset_edge_attributes(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]

    def _update_edge_attributes(self, state):
        """
        Update the edge attributes latency and remaining capacity along self.state.
        """
        service_placement, sp_indices, quantities = self._state_to_vectors(state)

        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            path = self._select_shortest_path(origin, target, sp_indices[demand_index])
            demand_path = path_node_to_path_tuple(path)

            for edge in demand_path:
                self.graph.edges[edge]["remaining_capacity"] -= quantities[demand_index]
                self.graph.edges[edge]["latency"] += (
                    quantities[demand_index] * self.graph.edges[edge]["alpha"]
                )

    def _reset_node_attributes(self):
        for node in self.graph.nodes:
            # Update remaining resources
            self.graph.nodes[node]["remaining_resources"] = self.graph.nodes[node][
                "available_resources"
            ]

    def _update_node_attributes(self, state):
        service_placement = self._state_to_vectors(state)[0]

        for service_index in range(self.problem_instance.number_of_service):
            target_node = service_placement[service_index]
            required_resources = self.problem_instance.needed_resources[service_index]
            self.graph.nodes[target_node]["remaining_resources"] -= required_resources

    def _test_all_constraints(self, state: list) -> bool:
        """
        Une fois que le graphe est mis à jour avec remaining_capacity,
        latency, remaining_resources, on s'assure que les contraintes
        sont respectées.
        """

        service_placement, sp_indices, quantities = self._state_to_vectors(state)

        for edge in self.graph.edges:  # Capacité pour chaque arc
            if self.graph.edges[edge]["remaining_capacity"] < 0.0:  # Edge capacity
                return False

        for demand_index in range(
            self.problem_instance.total_number_of_demands
        ):  # Pour chaque demande, quantité et latence limite
            if (
                quantities[demand_index] > self.problem_instance.max_flows[demand_index]
            ):  # Max flows
                return False

            service_index = self.problem_instance.demand_to_service[
                demand_index
            ]  # Latency
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            demand_path = path_node_to_path_tuple(
                self._select_shortest_path(origin, target, sp_indices[demand_index])
            )
            latency = sum(self.graph.edges[edge]["latency"] for edge in demand_path)
            if latency > self.problem_instance.latencies[demand_index]:
                return False

        for node in self.graph.nodes:  # Ressource pour chaque noeud
            if self.graph.nodes[node]["remaining_resources"] < 0.0:
                return False

        return True

    def _test_solution(self, state: list) -> bool:
        """
        Test if a given solution self.state is doable or not.
        - Capacity conditions
        - Latency conditions
        - Max flow conditions
        - Resources conditions
        """
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes(state)
        self._update_node_attributes(state)
        return self._test_all_constraints(state)

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
        # x latences
        self.action_space = MultiDiscrete(
            [
                3,  # Déplacer service, changer PCC ou change Q
                self.problem_instance.number_of_service,  # Choisir un service
                self.graph.number_of_nodes(),  # Le placer ici
                self.problem_instance.total_number_of_demands,  # Choisir une demande
                3,  # Augmenter ou diminuer sa quantité
                self.max_pcc,  # Choisir l'indice du PCC
            ]
        )

        # State : Ou sont placés les services (taille S, max n)
        #       x Quel est l'indice du PCC pour chaque demande (taille d, max sp_number)
        #       x Quantité de chaque demande (taille d, max q_dmax)

        high = max(
            self.graph.number_of_nodes(),
            self.max_pcc,
            self.problem_instance.max_flows[0],
        )
        observation_shape = (
            self.problem_instance.number_of_service  # Placement pour chaque service
            + self.problem_instance.total_number_of_demands  # PCC indices pour chaque demande
            + self.problem_instance.total_number_of_demands  # Flow pour chaque demande
            + self.problem_instance.total_number_of_demands  # Latence en proportion pour chaque demande
            + self.graph.number_of_edges()  # Capacité de chaque arc en proportion
            + self.graph.number_of_nodes()  # Ressources de chaque noeud
            + self.problem_instance.total_number_of_demands  # Flow max
            + self.problem_instance.total_number_of_demands  # Latences max
            + self.graph.number_of_edges()  # Capacité de chaque arc
            + self.graph.number_of_nodes()  # Ressources disponibles pour chaque serveur
            + self.problem_instance.number_of_service  # Ressources nécessaires par service
            + self.problem_instance.total_number_of_demands  # Longueur de chaque chemin utilisé
        )
        low_array = np.zeros(observation_shape)
        high_array = high * np.ones(observation_shape)
        self.observation_space = Box(low=low_array, high=high_array)

    def _max_score_ever(self) -> float:
        return max(self.scores)

    def _max_score_episode(self) -> float:
        return max(self.scores[-self.steps_done :])

    def _best_solution_ever(self) -> tuple:
        index = np.argmax(self.scores)
        return self.solutions[index]

    def _state_to_variables(self, state: list) -> tuple:
        """
        State to (x, y, q).
        """
        # State : Ou sont placés les services (taille S, max n)
        #       x Quel est le numéro du PCC pour chaque demande (taille d, max sp_number)
        #       x Quantité de chaque demande (taille d, max q_dmax)
        service_placement, sp_indices, choosen_quantities = self._state_to_vectors(
            state
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

    def _state_to_vectors(self, state: np.ndarray) -> tuple:
        """
        State to (service_placement, sp_indices, quantities)
        """
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands
        service_placement = state[:S]
        sp_indices = state[S : S + D]
        choosen_quantities = state[S + D : S + 2 * D]
        return service_placement, sp_indices, choosen_quantities

    def _observe(self, state: list) -> np.ndarray:
        """
        Transform a given state into the observation vector.
        """
        self._reset_edge_attributes()
        self._reset_node_attributes()
        self._update_edge_attributes(state)
        self._update_node_attributes(state)

        latence_proportion = [
            self._get_latency_demand(demand_index, state)
            / self.problem_instance.latencies[demand_index]
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]

        arc_saturation_proportion = [
            self.graph.edges[edge]["remaining_capacity"]
            / self.graph.edges[edge]["capacity"]
            for edge in self.graph.edges
        ]

        server_saturation = [
            self.graph.nodes[node]["remaining_resources"]
            / self.graph.nodes[node]["available_resources"]
            for node in self.graph.nodes
        ]

        longueur_chemins = [
            len(self._get_path_demand(demand_index, state))
            for demand_index in range(self.problem_instance.total_number_of_demands)
        ]

        return np.array(
            state
            + latence_proportion
            + arc_saturation_proportion
            + server_saturation
            + self.problem_instance.max_flows
            + self.problem_instance.latencies
            + self.problem_instance.network_graph.capacities
            + self.problem_instance.network_graph.available_resources
            + self.problem_instance.needed_resources
            + longueur_chemins
        )

    def _get_latency_demand(self, demand_index: int, state: list) -> float:
        """For a given demand, get its associated latency."""
        path = self._get_path_demand(demand_index, state)
        demand_path = path_node_to_path_tuple(path)
        return sum(self.graph.edges[edge]["latency"] for edge in demand_path)

    def _get_path_demand(self, demand_index: int, state: list) -> list:
        service_placement, sp_indices, _ = self._state_to_vectors(state)
        service_index = self.problem_instance.demand_to_service[demand_index]
        origin = self.problem_instance.origins[demand_index]
        target = service_placement[service_index]
        return self._select_shortest_path(origin, target, sp_indices[demand_index])
