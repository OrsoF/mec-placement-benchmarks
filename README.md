# Edge Computing Service Placement

Research code for **Faster Latency-Constrained Service Placement in Edge
Computing with Deep Reinforcement Learning**, IFIP Networking 2025.

The repository implements the Service Placement Problem (SPP) experiments from
the paper: exact arc ILP, structure-aware heuristics, CMA-ES, PPO-based
reinforcement learning, and a two-agent hierarchical PPO variant on synthetic
edge/cloud topologies. The code is available as the installable `mecbench` Python package.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

On Linux/macOS, use `source .venv/bin/activate`. Exact solvers require a working
Gurobi license visible to Pyomo; install the `gurobi` extra for its Python binding.
Optional extras: `benchmarks`, `exact`, `gurobi`, `evolutionary`,
`environments`, `rl`, `test`, and `dev`. `requirements.txt` installs the runtime
and test extras.
The base package needs only NumPy and NetworkX.

Install `.[test]` to run tests with `python -m pytest` and build distributions
with `python -m build`. Tests that need absent optional dependencies are explicitly
skipped; failures importing an installed dependency fail the suite.
For development, install `.[dev]`, then run `python -m ruff check src tests examples`,
`python -m ruff format --check src tests examples`, and `python -m mypy`.
See [the development guide](docs/development.md) for CI profiles and typing scope.

## Quickstart

```python
from mecbench import make_tree_instance, solve, validate

problem = make_tree_instance(n_users=20, seed=42)
result = solve(problem, solver="fast_heuristic", seed=42)
report = validate(problem, result.solution)
print(result.objective, report.feasible)  # 140.0 True
```

Run `python examples/quickstart.py` or `mecbench --seed 42` after installation.
The CLI defaults to one tree instance, the fast heuristic and CSV output.
`solve` validates every candidate, returns its status and detailed violations,
and leaves the objective empty for invalid candidates. See [the API guide](docs/api.md)
for solver options, optional installations, reproducibility and error handling.

## Main Experiments

`mecbench` reproduces the tree and star benchmark sweeps. It appends rows to
`artifacts/data/*.csv` and writes objective/runtime plots under
`artifacts/figures/`.

```bash
mecbench --instance tree --solvers ilp_arc heuristic fast_heuristic cma ppo hrl --seed 42 --start-users 20 --stop-users 410 --step-users 10 --plots
mecbench --instance star --solvers ilp_arc heuristic fast_heuristic cma ppo hrl --seed 42 --groups 4 --start-users 15 --stop-users 125 --step-users 5 --plots
```

`python -m mecbench` invokes the same CLI. Use `--instance tree` or `--instance star`
to select the experiment topology.

The tree experiment builds instances with `create_tree_problem_instance(n_users,
neighbors)`: users are connected locally, then to one wireless access point and
one cloud node.

The star experiment builds instances with `create_star_problem_instance(n_groups,
users_per_group, neighbors)`: users are split into local groups, each group has
one wireless access point, and all groups share the cloud.

Common options:

- `--instance`: `tree` or `star`.
- `--solvers`: registered names; common choices are `ilp_arc`, `heuristic`, `fast_heuristic`, `cma`, `ppo`, `hrl`. Use `--help` for all names.
- `--seed`: search seed, recorded in CSV (default 0).
- `--neighbors`: local user-neighborhood parameter.
- `--ilp-time-limit`: Gurobi time limit for `ilp_arc`.
- `--output`: CSV path.
- `--figures-dir`: output directory for plots.
- `--plots`: generate figures (requires the `benchmarks` extra).
- `--no-plots`: append CSV rows without figures (default).

CSV schema:

```text
instance,solver,users,neighbors,groups,score,runtime,buildtime,status,feasible,seed,error
```

An invalid candidate has an empty score and its violations in `error`. Existing
CSV files with an older header require a new `--output` path.

Default figures:

- `artifacts/figures/tree/tree_objective_vs_graph_size.pdf`
- `artifacts/figures/tree/tree_runtime_vs_graph_size.pdf`
- `artifacts/figures/star/star_objective_vs_graph_size.pdf`
- `artifacts/figures/star/star_runtime_vs_graph_size.pdf`

## Code Structure

```text
src/mecbench/domain/        Problem objects, solutions, reference validator
src/mecbench/compat/        Historical problem-state and experiment adapters
src/mecbench/instances/     Synthetic builders and topology loading
src/mecbench/solvers/       One file per solver, without family subfolders
src/mecbench/utils/         Shared contracts, registry, workspace and solver helpers
src/mecbench/environments/  Gymnasium environments and RL support code
src/mecbench/benchmarks/    Runner, metrics, reporting, result I/O
src/mecbench/cli.py         Command-line entry point

tests/unit/                Constraint and variable-domain tests
tests/integration/         Builders, component integration, package and CLI
tests/regression/          Shared-load regressions and MILP comparison
examples/                  Executable research examples
docs/                      Architecture, migration notes, research document
data/                      Input topologies
artifacts/                 Generated results, figures, and trained agents
archive/                   Historical scripts and notebooks
```

Solver modules expose a `Solver` class. Their constructors explicitly call
`GenericSolver.__init__` to initialize `x_var`, `y_var`, `q_var`, `score`,
`runtime`, `build_time`, and `termination_condition`. Array outputs and metrics
are populated by `run()`. Shared support lives in `utils/`: `solver_base`,
`solver_registry`, `solver_workspace`, `solver_paths`, `heuristic_solvers`,
`ilp_solvers`, `rl_solvers`, `arc_formulation`, and `path_formulation`.

Feasibility checks use `mecbench.domain.validation.validate_arc_solution(problem, x, y, q)`.
It raises `SolutionValidationError` (`ValueError`) for invalid dimensions,
non-finite values, nonbinary routing/placement, demand bounds, placement,
resources, capacities, flow conservation, or latency. Quantities may be
continuous, as in the arc MILP. Each arc's latency uses its total traffic
`beta[a] + alpha[a] * sum(x[d, a] * q[d] for d in demands)`. The default absolute
tolerance is `1e-6`. Compact solutions are converted to arc variables before
the same checks; fractional placement and path indices are rejected.

Run the reference tests with `python -m pytest tests/unit tests/integration tests/regression`.
They include shared-arc counterexamples and an exhaustive comparison against
the Pyomo arc constraints without requiring Gurobi. Those comparison tests
require Pyomo and are skipped if it is unavailable. Balanced routing cycles
are allowed by flow conservation, matching the arc MILP.

## Docker

```bash
docker build -t mecbench .
docker run --rm mecbench --help
docker run --rm -v "${PWD}/artifacts:/app/artifacts" mecbench --instance tree --solvers fast_heuristic --start-users 2 --stop-users 3 --neighbors 0 --no-plots
```

The default image includes benchmark/reporting dependencies. To include exact
solvers, build with `--build-arg MECBENCH_EXTRAS=benchmarks,exact,gurobi`; configure
your Gurobi license in the container. RL and evolutionary extras are available
through the same build argument.

See [architecture and migration](docs/architecture.md) for the import mapping.
This migration changes pickle class paths; regenerate historical instance
pickles before loading them with the new namespace. The project is licensed
under [MIT](LICENSE).
