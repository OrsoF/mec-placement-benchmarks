"""Dependency declarations for package import smoke tests."""


def module_requirements(module: str) -> tuple[str, ...]:
    if module == "mecbench.solvers.exact.extraction":
        return ()
    if module.startswith("mecbench.solvers.rl."):
        return ("stable_baselines3",)
    if module.startswith(
        ("mecbench.solvers.exact.", "mecbench.solvers.relaxations.")
    ) or module in {
        "mecbench.solvers.arc_formulation",
        "mecbench.solvers.path_formulation",
    }:
        return ("pyomo",)
    if module == "mecbench.solvers.evolutionary.cma":
        return ("cma",)
    if module in {
        "mecbench.environments.spp_env",
        "mecbench.environments.hrl_env",
        "mecbench.environments.observation",
    }:
        return ("gymnasium",)
    if module == "mecbench.benchmarks.io":
        return ("pandas", "tqdm")
    if module == "mecbench.benchmarks.reporting":
        return ("matplotlib",)
    return ()
