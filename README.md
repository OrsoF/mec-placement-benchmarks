# Edge Computing Service Placement

Research code for **Faster Latency-Constrained Service Placement in Edge
Computing with Deep Reinforcement Learning**, IFIP Networking 2025.

The repository implements the Service Placement Problem (SPP) experiments from
the paper: exact arc ILP, structure-aware heuristics, CMA-ES, PPO-based
reinforcement learning, and a two-agent hierarchical PPO variant on synthetic
edge/cloud topologies. It is a research workbench, not a packaged library.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, use `source .venv/bin/activate`. Exact solvers require a working
Gurobi license visible to Pyomo.

## Main Experiments

`exp.py` reproduces the tree and star benchmark sweeps. It appends rows to
`artifacts/data/*.csv` and writes objective/runtime plots under
`artifacts/figures/`.

```bash
python exp.py --instance tree --solvers ilp_arc heuristic fast_heuristic cma PPO HRL --start-users 20 --stop-users 410 --step-users 10
python exp.py --instance star --solvers ilp_arc heuristic fast_heuristic cma PPO HRL --groups 4 --start-users 15 --stop-users 125 --step-users 5
```

`exp_tree.py` and `exp_star.py` remain compatibility wrappers around `exp.py`.

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
core/          Problem classes, graph utilities, and solver base contract
instances/     Tree, star, GML, and larger synthetic instance builders
solvers/       Exact, relaxation, heuristic, evolutionary, and RL solvers
rl/            Gymnasium SPP environment and RL support code
data_io/       Pickle and experiment-result I/O
plotting/      Plot helpers for aggregate results and solutions
experiments/   Small exploratory scripts
data/          Input topologies
artifacts/     Generated CSVs, figures, instances, and trained agents
archive/       Legacy scripts and notebooks kept for reference
```

Solver modules expose a `Solver` class with the usual output fields after
`run()`: `x_var`, `y_var`, `q_var`, `score`, `runtime`, and, when available,
`build_time`.
