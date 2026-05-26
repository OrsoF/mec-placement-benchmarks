import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("torch_geometric")

from utils.build_instance_parallel import build_star_instance
from utils.gym_instance import SPPEnv


def test_gym_environment_reset_and_random_steps():
    instance = build_star_instance(
        number_of_groups=1,
        users_per_group=3,
        neighbors_connection=1,
    )
    env = SPPEnv(
        instance,
        max_step=3,
        max_pcc=2,
        warm_start_heuristic=False,
        observation_type="vector",
    )

    observation, info = env.reset()

    assert isinstance(info, dict)
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()

    for step_index in range(3):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        assert observation.shape == env.observation_space.shape
        assert np.isfinite(observation).all()
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if step_index < 2:
            assert not truncated
        else:
            assert truncated
