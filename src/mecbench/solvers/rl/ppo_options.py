"""PPO reinforcement-learning solver with best-state reset reuse."""

from mecbench.solvers.rl.base_solver import PPOOptionsSolver


class Solver(PPOOptionsSolver):
    """Compatibility entry point for the PPO options solver."""
