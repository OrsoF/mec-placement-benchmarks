import pytest


@pytest.mark.integration
@pytest.mark.requires_rl
def test_rl_solver_can_train_for_one_timestep(small_tree_instance):
    pytest.importorskip("stable_baselines3")

    from rl.base_solver import Solver as RLSolver

    from rl.spp_env import SPPEnv
    from stable_baselines3 import A2C

    env = SPPEnv(
        main_problem_instance=small_tree_instance,
        max_step=2,
        warm_start_heuristic=False,
    )
    model = A2C("MlpPolicy", env, verbose=0, seed=0, n_steps=2)

    model.learn(total_timesteps=1, progress_bar=False)

    assert max(env.served_quantity_history) >= sum(small_tree_instance.min_flows)


@pytest.mark.integration
@pytest.mark.requires_cma
def test_cma_dependency_is_available_for_cma_workflows():
    pytest.importorskip("cma")


@pytest.mark.integration
@pytest.mark.requires_pyomo
def test_pyomo_dependency_is_available_for_exact_solver_workflows():
    pytest.importorskip("pyomo")
