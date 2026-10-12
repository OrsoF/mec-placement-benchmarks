# Edge Computing Service Placement

Research code for **Faster Latency-Constrained Service Placement in Edge
Computing with Deep Reinforcement Learning**, IFIP Networking 2025.

The `mecbench` Python package provides service placement instances, exact
solvers, heuristics, evolutionary search and reinforcement learning baselines.

## Install

Python 3.10+ is required. The base package depends on NumPy and NetworkX.

```bash
python -m pip install -e .
```

Exact, evolutionary and RL solvers use optional dependencies; exact solvers
also require a Gurobi license. See [solver installations](docs/api.md#solving).

## Quickstart

```python
from mecbench import make_tree_instance, solve, validate

problem = make_tree_instance(n_users=20, seed=42)
result = solve(problem, solver="fast_heuristic", seed=42)
report = validate(problem, result.solution)
print(result.objective, report.feasible)  # 140.0 True
```

`solve()` validates each candidate and returns its objective and status.

## Command line

```bash
mecbench --seed 42
mecbench --help
```

The default command solves one tree instance and writes CSV results.

## Run all tests

From the repository root, install all test dependencies, including RL support:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[benchmarks,exact,environments,evolutionary,rl,test]"
python -m pytest --require-extras -ra
```

This runs the full suite, including tests otherwise skipped for missing optional
dependencies. `--require-extras` makes missing dependencies fail instead of skip;
`-ra` shows the reasons for any non-passing tests. No GPU or Gurobi license is
required for the current suite. See [development](docs/development.md) for other profiles.

## Guides

- [API, solvers and validation](docs/api.md)
- [Experiments and Docker](docs/experiments.md)
- [Development and tests](docs/development.md)
- [Architecture and migration](docs/architecture.md)
- [Research directions](docs/research.md)

Licensed under [MIT](LICENSE).
