import gymnasium as gym
import networkx as nx
import numpy as np

from core.graph_utils import path_node_to_path_tuple
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.solving_instance import SolvingInstance
from rl.observation import (
    make_action_space,
    make_vector_observation,
    make_vector_observation_space,
)
from rl.path_cache import ShortestPathCache
from rl.reward import served_quantity_score
from rl.state_codec import (
    clip_state_to_bounds,
    make_basic_state,
    state_to_variables,
    state_to_vectors,
)
from solvers.heuristic.heuristicprecise import Solver

rng = np.random.default_rng(0)


class SPPEnv(gym.Env):
    def __init__(
        self,
        main_problem_instance: ProblemInstance,
        max_step: int = 1000,
        max_path_candidates: int = 3,
        training_set: list[ProblemInstance] = None,
        warm_start_heuristic: bool = True,
        observation_type: str = "vector",
        reuse_best_on_reset: bool = False,
    ):
        self.max_path_candidates = max_path_candidates
        self.max_steps = max_step
        self.training_instances = training_set
        self.warm_start_heuristic = warm_start_heuristic
        self.observation_type = observation_type
        assert self.observation_type in [
            "vector",
        ], "Only vector observations are implemented."
        self.reuse_best_on_reset = reuse_best_on_reset
        self.base_problem_instance = main_problem_instance
        self.served_quantity_history = [0]
        self.steps_done = 0
        self.episode_start_score_index = 0
        self._load_problem_instance(main_problem_instance)
        self.best_state_record = (self.state, self._score_state(self.state))

    def _load_problem_instance(self, problem_instance: ProblemInstance):
        """Load problem-specific objects without rebuilding environment config."""
        self.problem_instance = problem_instance
        self.network_graph = self.problem_instance.network_graph
        self.graph = self.network_graph.nx_graph
        self.path_cache = ShortestPathCache(self.graph)
        self.saved_shortest_path = self.path_cache._paths
        self.solution_checker = SolvingInstance(self.problem_instance)
        self.initial_solution = None
        self._configure_spaces()
        self.state = self._make_initial_state()
        self.best_state_record = (self.state, self._score_state(self.state))

    def _make_initial_state(self) -> np.ndarray:
        """Build the initial placement, routing, and quantity state."""
        if self.warm_start_heuristic and self.initial_solution is not None:
            return self.initial_solution

        basic_state = self._make_basic_state()

        if self.warm_start_heuristic:
            heuristic_solution = Solver(self.problem_instance, 3, 3)
            heuristic_solution.run()

            self.initial_solution = np.concatenate(
                (
                    np.asarray(heuristic_solution.placement, dtype=int),
                    np.asarray(heuristic_solution.routing, dtype=int),
                    np.asarray(heuristic_solution.quantities, dtype=int),
                )
            )
            self.served_quantity_history.append(sum(heuristic_solution.quantities))
            if self._is_feasible_state(self.initial_solution):
                return self.initial_solution
            self.initial_solution = None

        return basic_state

    def _make_basic_state(self) -> np.ndarray:
        return make_basic_state(self.problem_instance, self.graph)

    def reset(
        self,
        seed: float = 0,
        initial_instance: bool = False,
        warm_start_heuristic: bool | None = None,
        observation_type: str | None = None,
        reuse_best_on_reset: bool | None = None,
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
        if reuse_best_on_reset is None:
            reuse_best_on_reset = self.reuse_best_on_reset
        self.warm_start_heuristic = warm_start_heuristic
        self.observation_type = observation_type
        self.reuse_best_on_reset = reuse_best_on_reset

        next_problem_instance = self.problem_instance

        if self.training_instances is not None:
            next_problem_instance = rng.choice(self.training_instances)

        if initial_instance:
            next_problem_instance = self.base_problem_instance

        if next_problem_instance is not self.problem_instance:
            self._load_problem_instance(next_problem_instance)
        else:
            self._configure_spaces()

        if not self.reuse_best_on_reset:
            self.state = self._make_initial_state()
        else:
            self.state = self.best_state_record[0]

        self.steps_done = 0
        self.episode_start_score_index = len(self.served_quantity_history)
        return self._observe(self.state), {}

    def _try_state_update(self, state, update_fn) -> tuple:
        """Apply one state mutation and keep the old state when infeasible."""
        new_state = np.asarray(state).copy()
        update_fn(*self._state_to_vectors(new_state))
        is_feasible = self._is_feasible_state(new_state)
        if is_feasible:
            return new_state, is_feasible
        return state, is_feasible

    def _update_placement(
        self, state, selected_service: int, chosen_node: int
    ) -> tuple:
        """Try moving one service, keeping the old state when infeasible."""
        def move_service(placement, _, __):
            placement[selected_service] = chosen_node

        return self._try_state_update(state, move_service)

    def _update_route(self, state, selected_demand: int, chosen_path_index: int) -> tuple:
        """Try rerouting one demand through the selected path index."""
        def reroute_demand(_, routing, __):
            routing[selected_demand] = chosen_path_index

        return self._try_state_update(state, reroute_demand)

    def _update_quantity(
        self, state, selected_demand: int, quantity_action: int
    ) -> tuple:
        """For one demand, apply -1, 0, or +1 according to quantity_action."""
        quantity_delta = quantity_action - 1

        def resize_demand(_, __, quantities):
            quantities[selected_demand] += quantity_delta

        return self._try_state_update(state, resize_demand)

    def render(self):
        print(self.state)

    def step(self, action: tuple):
        terminated = False
        info = {}

        state = self.state

        # Action layout: [operation, service, node, demand, quantity delta, path index].
        (
            operation,
            selected_service,
            chosen_node_for_this_service,
            selected_demand,
            quantity_action,
            chosen_path_index,
        ) = action

        if operation == 0:
            state, done = self._update_placement(
                state, selected_service, chosen_node_for_this_service
            )
            reward = -1 if done else -10
        elif operation == 1:
            state, done = self._update_route(state, selected_demand, chosen_path_index)
            reward = -1 if done else -2
        else:
            state, _ = self._update_quantity(state, selected_demand, quantity_action)
            current_served_quantity = sum(self._state_to_vectors(state)[2])
            best_episode_quantity = self._best_episode_score()
            reward = max(0, current_served_quantity - best_episode_quantity)

        self.state = state
        observation = self._observe(state)
        self.served_quantity_history.append(sum(self._state_to_vectors(self.state)[2]))

        current_score = self._score_state(state)
        if self.best_state_record[1] < current_score:
            self.best_state_record = (state, current_score)

        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps
        return observation, reward / 100, terminated, truncated, info

    def _get_path_candidate(
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
            path = self._get_path_candidate(
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

    def _is_feasible_state(self, state=None) -> bool:
        """Test if a state is feasible."""
        if state is None:
            state = self.state
        placement, routing, quantities = self._state_to_vectors(state)
        for demand_index in range(self.problem_instance.total_number_of_demands):
            service_index = self.problem_instance.demand_to_service[demand_index]
            origin = self.problem_instance.origins[demand_index]
            target = placement[service_index]
            try:
                self._get_path_candidate(origin, target, routing[demand_index])
            except (IndexError, nx.NetworkXException):
                return False
        return self.solution_checker.test_solution(placement, routing, quantities)

    def _configure_spaces(self):
        """Build the Gym action and observation spaces."""
        if self.observation_type == "vector":
            # State layout: [service placements | path indices | demand quantities].
            self.action_space = make_action_space(
                self.problem_instance, self.graph, self.max_path_candidates
            )
            self.observation_space = make_vector_observation_space(
                self.problem_instance, self.graph, self.max_path_candidates
            )
        else:
            assert False, "Unknown observation space."

    def clip_state(self):
        """Clip the current flat state to action-compatible integer bounds."""
        self.state = clip_state_to_bounds(
            self.problem_instance,
            self.graph,
            self.max_path_candidates,
            self.state,
        )
        return self.state

    def _best_episode_score(self) -> float:
        episode_scores = self.served_quantity_history[self.episode_start_score_index :]
        return max(episode_scores) if episode_scores else 0

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
        return make_vector_observation(self, state)

    def _get_demand_latency(self, demand_index: int, state: list) -> float:
        """For a given demand, get its associated latency."""
        path = self._get_demand_path(demand_index, state)
        demand_path = path_node_to_path_tuple(path)
        return sum(self.graph.edges[edge]["latency"] for edge in demand_path)

    def _get_demand_path(self, demand_index: int, state: list) -> list:
        service_placement, path_indices, _ = self._state_to_vectors(state)
        service_index = self.problem_instance.demand_to_service[demand_index]
        origin = self.problem_instance.origins[demand_index]
        target = service_placement[service_index]
        return self._get_path_candidate(
            origin, target, path_indices[demand_index]
        )

    def _score_state(self, state):
        return served_quantity_score(self.problem_instance, state)
