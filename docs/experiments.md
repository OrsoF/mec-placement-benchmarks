# Benchmark experiments

Run commands from the repository root after installing `mecbench`.
For the full solver sweeps below, install the optional dependencies:

```bash
python -m pip install -e ".[benchmarks,exact,gurobi,evolutionary,rl]"
```

Exact solvers require a working Gurobi license. See the [API guide](api.md)
for installation options by solver and [development](development.md) for tests.

## Tree and star sweeps

`mecbench` reproduces the tree and star benchmark sweeps. It appends rows to
`artifacts/data/*.csv` and writes objective/runtime plots under
`artifacts/figures/`.

```bash
mecbench --instance tree --solvers ilp_arc heuristic fast_heuristic cma_es ppo hierarchical_ppo --seed 42 --start-users 20 --stop-users 410 --step-users 10 --plots
mecbench --instance star --solvers ilp_arc heuristic fast_heuristic cma_es ppo hierarchical_ppo --seed 42 --groups 4 --start-users 15 --stop-users 125 --step-users 5 --plots
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
- `--solvers`: registered names; common choices are `ilp_arc`, `heuristic`, `fast_heuristic`, `cma_es`, `ppo`, `hierarchical_ppo`. Use `--help` for all names.
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
