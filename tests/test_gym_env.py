from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from instances.parallel_builder import build_star_instance
from instances.tree_builder import make_tree_instance
from rl.spp_env import SPPEnv


def make_env(max_step=3):
    instance = build_star_instance(
        number_of_groups=1,
        users_per_group=3,
        neighbors_connection=1,
    )
    return SPPEnv(
        instance,
        max_step=max_step,
        max_pcc=2,
        warm_start_heuristic=False,
        observation_type="vector",
    )


def make_tree_env(max_step=3):
    instance = make_tree_instance(number_of_users=2, neighbors_connection=0)
    return SPPEnv(
        instance,
        max_step=max_step,
        max_pcc=2,
        warm_start_heuristic=False,
        observation_type="vector",
    )


def test_primary_gym_environment_lives_in_rl_package():
    root_dir = Path(__file__).resolve().parents[1]

    assert (root_dir / "rl" / "spp_env.py").exists()
    assert not (root_dir / "utils").exists()
    assert (root_dir / "archive" / "legacy" / "gym_instance_2.py").exists()


def test_gym_environment_reset_and_random_steps():
    env = make_env()

    observation, info = env.reset()

    assert isinstance(info, dict)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert env.observation_space.dtype == np.float32
    assert np.isfinite(observation).all()

    for step_index in range(3):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if step_index < 2:
            assert not truncated
        else:
            assert truncated


def test_gym_state_to_vectors_splits_state_by_problem_dimensions():
    env = make_env()
    state = env._basic_placement_state()

    placement, routing, quantities = env._state_to_vectors(state)

    assert len(placement) == env.problem_instance.number_of_service
    assert len(routing) == env.problem_instance.total_number_of_demands
    assert len(quantities) == env.problem_instance.total_number_of_demands
    assert list(quantities) == env.problem_instance.min_flows


def test_gym_score_uses_quantities_only():
    env = make_env()
    state = env._basic_placement_state()

    placement, routing, quantities = env._state_to_vectors(state)

    assert env._get_score(state) == sum(quantities)
    assert env._get_score(state) != sum(placement) + sum(routing)


def test_gym_state_to_variables_returns_feasible_solution_arrays():
    env = make_tree_env()
    state = env._basic_placement_state()

    x_var, y_var, q_var = env._state_to_variables(state)

    assert x_var.shape == (
        env.problem_instance.total_number_of_demands,
        env.graph.number_of_edges(),
    )
    assert y_var.shape == (
        env.problem_instance.number_of_service,
        env.graph.number_of_nodes(),
    )
    assert q_var.shape == (env.problem_instance.total_number_of_demands,)
    assert env._test_solution(state)


def test_gym_rejects_unimplemented_gcn_observation_mode():
    instance = make_tree_instance(number_of_users=2, neighbors_connection=0)

    with pytest.raises(AssertionError, match="Only vector observations"):
        SPPEnv(instance, max_step=2, observation_type="gcn")


def test_gym_reset_preserves_or_accepts_reset_configuration():
    first_instance = make_tree_instance(number_of_users=2, neighbors_connection=0)
    second_instance = make_tree_instance(number_of_users=3, neighbors_connection=0)
    env = SPPEnv(
        first_instance,
        max_step=2,
        training_set=[second_instance],
        warm_start_heuristic=False,
        observation_type="vector",
        options=True,
    )

    env.reset(warm_start_heuristic=False, observation_type="vector", options=True)

    assert env.warm_start_heuristic is False
    assert env.observation_type == "vector"
    assert env.options is True


def test_gym_invalid_path_index_is_infeasible():
    env = make_tree_env()
    state = env._basic_placement_state()
    placement, routing, quantities = env._state_to_vectors(state)
    routing[0] = 999
    invalid_state = list(placement) + list(routing) + list(quantities)

    assert not env._test_solution(invalid_state)


def test_gym_reset_restores_step_count_and_initial_observation_shape():
    env = make_env(max_step=2)
    env.step(env.action_space.sample())

    observation, info = env.reset()

    assert env.steps_done == 0
    assert isinstance(info, dict)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32


def test_gym_truncates_at_max_step():
    env = make_env(max_step=1)

    _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    assert not terminated
    assert truncated
