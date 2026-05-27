import gymnasium as gym
import networkx as nx
import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.probleminstance import ProblemInstance
from core.problem_classes.solvinginstance import SolvingInstance
from rl.observation import make_action_space, make_vector_observation_space, observe_vector
from rl.path_cache import ShortestPathCache
from rl.reward import score_state
from rl.state_codec import state_to_variables, state_to_vectors
from solvers.heuristic.heuristicprecise import Solver

rnd_gen = np.random.default_rng(0)


class SPPEnv(gym.Env):
    def __init__(
        self,
        main_problem_instance: ProblemInstance,
        max_step: int,
        max_pcc: int = 3,
        training_set: list[ProblemInstance] = None,
        warm_start_heuristic: bool = True,
        observation_type: str = "vector",
        options: bool = False,
    ):
        self.problem_instance = main_problem_instance
        self.max_pcc = max_pcc
        self.max_step = max_step
        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph
        self.path_cache = ShortestPathCache(self.graph)
        self.saved_shortest_path = self.path_cache._paths
        self.list_of_instances = training_set
        self.warm_start_heuristic = warm_start_heuristic
        self.observation_type = observation_type
        assert self.observation_type in [
            "vector",
        ], "Only vector observations are implemented."
        self.options = options

        self.sol_ins = SolvingInstance(self.problem_instance)

        if not hasattr(self, "first_time"):
            self.initial_instance = main_problem_instance
        self.first_time = False

        self._make_action_observation_spaces()
        self.state = self._initial_state()
        self.best_solution = (self.state, self._get_score(self.state))

        self.steps_done = 0
        if not hasattr(self, "scores"):
            self.scores = [0]

    def _initial_state(self) -> list:
        """Build the initial placement, routing, and quantity state."""
        # State layout: [service placements | path indices | demand quantities].
        if hasattr(self, "lapremieresolution"):
            return self.lapremieresolution

        node_with_max_resources = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )
        service_placement = [
            node_with_max_resources
        ] * self.problem_instance.number_of_service
        path_indices = [0] * self.problem_instance.total_number_of_demands
        quantities = self.problem_instance.min_flows

        if not hasattr(self, "scores"):
            self.scores = [0]

        if self.warm_start_heuristic:
            heuristic_solution = Solver(self.problem_instance, 3, 3)
            heuristic_solution.run()

            service_placement = heuristic_solution.placement
            path_indices = heuristic_solution.routing
            quantities = heuristic_solution.quantities
            self.scores.append(sum(quantities))

            self.lapremieresolution = service_placement + path_indices + quantities

        return service_placement + path_indices + quantities

    def _basic_placement_state(self) -> list:
        node_with_max_resources = np.argmax(
            [self.graph.nodes[node]["available_resources"] for node in self.graph.nodes]
        )
        service_placement = [
            node_with_max_resources
        ] * self.problem_instance.number_of_service
        path_indices = [0] * self.problem_instance.total_number_of_demands
        quantities = self.problem_instance.min_flows
        return service_placement + path_indices + quantities

    def reset(
        self,
        seed: float = 0,
        initial_instance: bool = False,
        warm_start_heuristic: bool | None = None,
        observation_type: str | None = None,
        options: bool | None = None,
    ):
        """Reset environment, optionally overriding reset-time configuration."""
        super().reset(seed=seed)

        warm_start_heuristic = (
            self.warm_start_heuristic
            if warm_start_heuristic is None
            else warm_start_heuristic
        )
        observation_type = (
            self.observation_type if observation_type is None else observation_type
        )
        options = self.options if options is None else options

        if self.list_of_instances is not None:
            new_instance = rnd_gen.choice(self.list_of_instances)
            self.__init__(
                new_instance,
                self.max_step,
                self.max_pcc,
                self.list_of_instances,
                warm_start_heuristic=warm_start_heuristic,
                observation_type=observation_type,
                options=options,
            )

        if initial_instance:
            self.__init__(
                self.initial_instance,
                self.max_step,
                self.max_pcc,
                self.list_of_instances,
                warm_start_heuristic=warm_start_heuristic,
                observation_type=observation_type,
                options=options,
            )

        if not self.options:
            self.state = self._initial_state()
        else:
            self.state = self.best_solution[0]

        self.steps_done = 0
        return self._observe(self.state), {}

    def _update_placement(
        self, state: list, selected_service: int, choosen_node: int
    ) -> tuple:
        """Try moving one service, keeping the old state when infeasible."""
        new_state = state.copy()
        new_state[selected_service] = choosen_node
        done = self._test_solution(new_state)
        if done:
            return new_state, done
        return state, done

    def _reroute(self, state: list, selected_demand: int, choosen_sp: int) -> tuple:
        """Try rerouting one demand through the selected path index."""
        new_state = state.copy()
        new_state[self.problem_instance.number_of_service + selected_demand] = (
            choosen_sp
        )

        done = self._test_solution(new_state)
        if done:
            return new_state, done
        return state, done

    def _update_quantity(
        self, state: list, selected_demand: int, action_q: int
    ) -> tuple:
        """For one demand, apply -1, 0, or +1 according to action_q."""
        action_q = action_q - 1
        state_index = -self.problem_instance.total_number_of_demands + selected_demand
        new_state = state.copy()
        new_state[state_index] += action_q

        done = self._test_solution(new_state)
        if done:
            return new_state, done
        return state, done

    def render(self):
        print(self.state)

    def step(self, action: tuple):
        self.steps_done += 1

        truncated = self.steps_done >= self.max_step
        terminated = False
        info = {}

        if truncated:
            reward = 0.0
            return self._observe(self.state), reward, terminated, truncated, info

        state = self.state

        # Action layout: [operation, service, node, demand, quantity delta, path index].
        (
            operation,
            selected_service,
            choosen_node_for_this_service,
            selected_demand,
            action_q,
            choosen_sp,
        ) = action

        if operation == 0:
            state, done = self._update_placement(
                state, selected_service, choosen_node_for_this_service
            )
            reward = -1 if done else -10
        elif operation == 1:
            state, done = self._reroute(state, selected_demand, choosen_sp)
            reward = -1 if done else -2
        else:
            state, _ = self._update_quantity(state, selected_demand, action_q)
            current_q = sum(self._state_to_vectors(state)[2])
            best_q = self._max_score_episode()
            reward = max(0, current_q - best_q)

        self.state = state
        obs = self._observe(state)
        self.scores.append(sum(self._state_to_vectors(self.state)[2]))

        current_score = self._get_score(state)
        if self.best_solution[1] < current_score:
            self.best_solution = (state, current_score)

        return obs, reward / 100, terminated, truncated, info

    def _compute_nth_shortest_path(
        self, source: int, destination: int, index: int
    ) -> list:
        """Return the index-th shortest path from source to destination."""
        # Shortest-path generation is expensive, so route lookups are memoized.
        return self.path_cache.nth_shortest_path(source, destination, index)

    def _reset_edge_attributes(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]["remaining_capacity"] = self.graph.edges[edge][
                "capacity"
            ]
            self.graph.edges[edge]["latency"] = self.graph.edges[edge]["beta"]

    def _update_edge_attributes(self, state):
        """Update edge latency and remaining capacity for the current state."""
        service_placement, path_indices, quantities = self._state_to_vectors(state)

        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            path = self._compute_nth_shortest_path(
                origin, target, path_indices[demand_index]
            )
            demand_path = path_node_to_path_tuple(path)

            for edge in demand_path:
                self.graph.edges[edge]["remaining_capacity"] -= quantities[demand_index]
                self.graph.edges[edge]["latency"] += (
                    quantities[demand_index] * self.graph.edges[edge]["alpha"]
                )

    def _reset_node_attributes(self):
        for node in self.graph.nodes:
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
        """Check graph capacity, latency, flow, and resource constraints."""
        service_placement, path_indices, quantities = self._state_to_vectors(state)

        for edge in self.graph.edges:
            if self.graph.edges[edge]["remaining_capacity"] < 0.0:
                return False

        for demand_index in range(self.problem_instance.total_number_of_demands):
            if quantities[demand_index] > self.problem_instance.max_flows[demand_index]:
                return False

            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = service_placement[service_index]
            demand_path = path_node_to_path_tuple(
                self._compute_nth_shortest_path(
                    origin, target, path_indices[demand_index]
                )
            )
            latency = sum(self.graph.edges[edge]["latency"] for edge in demand_path)
            if latency > self.problem_instance.latencies[demand_index]:
                return False

        for node in self.graph.nodes:
            if self.graph.nodes[node]["remaining_resources"] < 0.0:
                return False

        return True

    def _test_solution(self, state: list) -> bool:
        """Test if a state is feasible."""
        placement, routing, quantities = self._state_to_vectors(state)
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            try:
                self._compute_nth_shortest_path(origin, target, routing[demand_index])
            except (IndexError, nx.NetworkXException):
                return False
        return self.sol_ins.test_solution(placement, routing, quantities)

    def _make_action_observation_spaces(self):
        """Build the Gym action and observation spaces."""
        if self.observation_type == "vector":
            # State layout: [service placements | path indices | demand quantities].
            self.action_space = make_action_space(
                self.problem_instance, self.graph, self.max_pcc
            )
            self.observation_space = make_vector_observation_space(
                self.problem_instance, self.graph, self.max_pcc
            )
        else:
            assert False, "Unknown observation space."

    def _max_score_ever(self) -> float:
        return max(self.scores)

    def _max_score_episode(self) -> float:
        return max(self.scores[-self.steps_done :])

    def _max_score_previous_episodes(self) -> float:
        return (
            max(self.scores[: -self.max_step])
            if len(self.scores) >= self.max_step
            else 0
        )

    def _state_to_variables(self, state: list) -> tuple:
        """Convert state to arc, placement, and quantity variables."""
        return state_to_variables(
            self.problem_instance,
            self.graph,
            self.network_graph,
            self.path_cache,
            state,
        )

    def _state_to_vectors(self, state: list) -> tuple:
        """Split state into service placement, path indices, and quantities."""
        return state_to_vectors(self.problem_instance, state)

    def _observe(self, state: list):
        if self.observation_type == "vector":
            return self._observe_vector(state)
        assert False, "Unknown observation space."

    def _observe_vector(self, state: list) -> np.ndarray:
        """Transform a state into the flat observation vector."""
        # Observation features depend on mutable graph capacities and resources.
        return observe_vector(self, state)

    def _get_latency_demand(self, demand_index: int, state: list) -> float:
        """For a given demand, get its associated latency."""
        path = self._get_path_demand(demand_index, state)
        demand_path = path_node_to_path_tuple(path)
        return sum(self.graph.edges[edge]["latency"] for edge in demand_path)

    def _get_path_demand(self, demand_index: int, state: list) -> list:
        service_placement, path_indices, _ = self._state_to_vectors(state)
        service_index = self.problem_instance.demand_to_service[demand_index]
        origin = self.problem_instance.origins[demand_index]
        target = service_placement[service_index]
        return self._compute_nth_shortest_path(
            origin, target, path_indices[demand_index]
        )

    def _get_score(self, state):
        return score_state(self.problem_instance, state)
