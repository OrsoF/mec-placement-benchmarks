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
python -m pip install -e ".[benchmarks,exact,test]"
```

On Linux/macOS, use `source .venv/bin/activate`. Exact solvers require a working
Gurobi license visible to Pyomo; install the `gurobi` extra for its Python binding.
Optional extras: `benchmarks`, `exact`, `gurobi`, `evolutionary`,
`environments`, `rl`, and `test`. `requirements.txt` installs all of them.
The base package needs only NumPy and NetworkX.

Run tests with `python -m pytest` and build a wheel with `python -m build`.

## Main Experiments

`mecbench` reproduces the tree and star benchmark sweeps. It appends rows to
`artifacts/data/*.csv` and writes objective/runtime plots under
`artifacts/figures/`.

```bash
mecbench --instance tree --solvers ilp_arc heuristic fast_heuristic cma PPO HRL --start-users 20 --stop-users 410 --step-users 10
mecbench --instance star --solvers ilp_arc heuristic fast_heuristic cma PPO HRL --groups 4 --start-users 15 --stop-users 125 --step-users 5
```

`python -m mecbench` invokes the same CLI. The historical commands are now
`examples/exp.py`, `examples/exp_tree.py`, and `examples/exp_star.py`; install
the package before running them.

`exp_tree.py` builds instances with `create_tree_problem_instance(n_users,
neighbors)`: users are connected locally, then to one wireless access point and
one cloud node.

`exp_star.py` builds instances with `create_star_problem_instance(n_groups,
users_per_group, neighbors)`: users are split into local groups, each group has
one wireless access point, and all groups share the cloud.

Common options:

- `--instance`: `tree` or `star`.
- `--solvers`: subset of `ilp_arc`, `heuristic`, `fast_heuristic`, `cma`, `PPO`, `HRL`.
- `--neighbors`: local user-neighborhood parameter.
- `--ilp-time-limit`: Gurobi time limit for `ilp_arc`.
- `--output`: CSV path.
- `--figures-dir`: output directory for plots.
- `--no-plots`: append CSV rows without regenerating figures.

CSV schema:

```text
instance,solver,users,neighbors,groups,score,runtime,buildtime
```

Default figures:

- `artifacts/figures/tree/tree_objective_vs_graph_size.pdf`
- `artifacts/figures/tree/tree_runtime_vs_graph_size.pdf`
- `artifacts/figures/star/star_objective_vs_graph_size.pdf`
- `artifacts/figures/star/star_runtime_vs_graph_size.pdf`

## Code Structure

```text
src/mecbench/domain/        Problem objects, solutions, reference validator
src/mecbench/instances/     Synthetic builders and topology loading
src/mecbench/solvers/       Base contract, heuristics, exact, evolutionary, RL
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

Solver modules expose a `Solver` class with the usual output fields after
`run()`: `x_var`, `y_var`, `q_var`, `score`, `runtime`, and, when available,
`build_time`.

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

## Research TODO

Main improvement axes extracted from `docs/suiteorange.tex`:

- Stabilize the simulation protocol: run short smoke sweeps before long
  campaigns, repeat experiments on several instances to smooth curves, and
  keep static and dynamic-instance evaluations clearly separated in result
  files and plots.
- Strengthen the dynamic setting: define a sequence-of-instances benchmark
  where solvers handle successive related SPP instances, then measure both cold
  starts and reuse of previous solutions or policies.
- Improve exact and linear-programming baselines: test whether Gurobi/Pyomo
  warm starts from previous, possibly infeasible, solutions help on dynamic
  instances; document negative results if warm starts are not useful.
- Extend hierarchical RL: implement a policy with actions triggered at
  different frequencies, e.g. frequent local actions plus slower placement or
  routing actions, with reward shaping that does not bias the objective.
- Audit reward shaping: compare the current shaped reward against neutral or
  uniformly penalized actions, and report the effect on score, feasibility,
  convergence speed, and optimality gap.
- Evaluate whether CMA-ES should remain a relaxed continuous policy or be
  adapted to integer actions; keep the integer variant only if the runtime and
  score tradeoff is convincing.
- Add transfer-learning experiments: reuse an RL policy across related static
  or dynamic instances and quantify when transfer improves convergence.
- Improve industrial relevance: build or import more realistic topologies,
  demands, latency budgets, and capacity distributions, ideally from Orange-like
  data when available.
- Make experimental outputs publication-ready: add reproducible config files,
  solver/version metadata, seeds, and scripts that regenerate the journal
  tables and figures from raw CSVs.
- Expand the bibliography and positioning: compare RL, ILP/LP, hierarchical
  network optimization, transfer learning, and industrial edge-placement papers
  to clarify the journal contribution.

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
