"""Dependency declarations for flat solver and utility import smoke tests."""

from mecbench.utils.solver_registry import REGISTRY


def is_rl_module(module: str) -> bool:
    return module == "mecbench.utils.rl_solvers" or any(
        spec.module == module and spec.extra == "rl" for spec in REGISTRY.values()
    )


def module_requirements(module: str) -> tuple[str, ...]:
    if is_rl_module(module):
        return ("stable_baselines3",)
    if module in {
        "mecbench.utils.arc_formulation",
        "mecbench.utils.arc_pyomo",
        "mecbench.utils.path_formulation",
    } or any(spec.module == module and spec.mathematical for spec in REGISTRY.values()):
        return ("pyomo",)
    if module == REGISTRY["cma_es"].module:
        return ("cma",)
    if module in {
        "mecbench.environments.spp_env",
        "mecbench.environments.hrl_env",
        "mecbench.environments.observation",
    }:
        return ("gymnasium",)
    if module == "mecbench.compat.experiment_io":
        return ("pandas",)
    if module == "mecbench.benchmarks.reporting":
        return ("matplotlib",)
    return ()
