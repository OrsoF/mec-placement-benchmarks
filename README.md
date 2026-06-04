# Edge Computing Service Placement

Research code for the service placement problem in edge-computing networks. The
project builds graph-based benchmark instances, represents service demands and
network constraints in compact Python objects, and compares exact optimization,
LP relaxation, heuristic, evolutionary, and reinforcement-learning solvers.

Given a directed symmetric network, the optimization task is to:

- place each service on one node with enough compute capacity;
- route every demand to the node hosting its requested service;
- choose served demand quantities between lower and upper bounds;
- satisfy edge capacities and latency limits;
- maximize total served quantity.

The code is a research workbench, not a packaged library. Most modules are meant
to be read and modified directly.

## Repository Layout

```text
.
|-- core/                   # Domain objects, graph utilities, solver contract
|-- instances/              # Synthetic, GML, and hierarchical instance builders
|-- solvers/                # Exact, relaxation, heuristic, evolutionary, RL solvers
|-- rl/                     # Gymnasium environment and RL state/reward helpers
|-- data_io/                # Pickle and experiment-result I/O
|-- plotting/               # Plotting utilities for results and solutions
|-- experiments/            # Exploratory experiment entry points
|-- data/topologies/        # GML topologies used by instance builders
|-- artifacts/              # Saved CSVs, figures, trained agents, generated data
|-- archive/legacy/         # Older scripts and notebooks kept for reference
|-- tests/                  # Unit, smoke, and optional integration tests
|-- run_experiments.py      # Batch experiment driver
|-- requirements.txt        # Python dependency list
|-- main.tex                # LaTeX result/table fragment
`-- README.md
```

## Core Model

The main domain objects live in `core/problem_classes/`.

- `NetworkGraph` wraps a NetworkX graph. Undirected inputs are converted to
  symmetric directed graphs, nodes are relabeled to integers, and node resources,
  arc capacities, and latency parameters are stored as graph attributes and flat
  lists.
- `Demand` stores an origin node, lower and upper flow bounds, and a maximum
  latency.
- `Service` groups one or more demands and a scalar resource requirement.
- `ServicePackage` groups services that share one network graph and have the
  same number of demands.
- `ProblemInstance` combines the graph and service package, extracts vectorized
  arrays used by solvers, and validates a baseline feasible solution.
- `SolvingInstance` converts compact placement/routing/quantity vectors to arc
  variables and tests candidate feasibility.

Solver outputs use the contract in `core/generic_solver.py`:

```python
x_var      # demand-by-arc routing incidence
y_var      # service-by-node placement incidence
q_var      # served quantity per demand
score      # objective value, usually sum(q_var)
runtime    # wall-clock solve time
build_time # model construction time when available
```

## Instance Builders

`instances/builder.py` exposes four current constructors:

```python
from instances.builder import (
    create_tree_problem_instance,
    create_star_problem_instance,
    create_gml_problem_instance,
    create_large_problem_instance,
)

instance = create_tree_problem_instance(n_users=10, neighbors=1)
```

The supported families are:

- `tree`: users connected locally, then to one wireless access point and one
  cloud node.
- `star`: grouped user meshes, one WAP per group, and a shared cloud node.
- `gml`: a topology read from `data/topologies/*.gml`, with random service
  origins.
- `large`: a hierarchical graph built by attaching edge and user layers to a
  supplied core graph.

Graph construction is in `instances/graph_build.py`; service-demand construction
is in `instances/package_build.py`.

## Solvers

Solver modules define a `Solver` class and generally inherit from
`GenericSolver` or an RL solver base.

- `solvers/exact/`: Pyomo/Gurobi arc and path MILP/ILP formulations, including
  heuristic and RL warm-start variants.
- `solvers/relaxations/`: LP relaxations of the arc and path formulations.
- `solvers/heuristic/`: local-search and precise heuristic solvers over compact
  placement, routing, and quantity vectors.
- `solvers/evolutionary/`: discrete and CMA-ES search over the same compact
  solution representation.
- `solvers/rl/`: PPO and A2C wrappers around the Gymnasium environment in
  `rl/spp_env.py`.

Minimal heuristic example:

```python
from instances.builder import create_tree_problem_instance
from solvers.heuristic.heuristic import Solver

problem = create_tree_problem_instance(n_users=8, neighbors=1)
solver = Solver(problem, seed=0, max_iter=200)
solver.run()

print(solver.score, solver.runtime)
problem._set_arc_solution(solver.x_var, solver.y_var, solver.q_var)
problem.check_current_solution()
```

Exact and relaxation solvers require a working Gurobi installation visible to
Pyomo. RL solvers require Stable-Baselines3, Gymnasium, and PyTorch.

## Setup

Use a fresh Python environment and install the listed dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

Optional dependencies have external requirements:

- `gurobipy` and Pyomo solver calls require a valid Gurobi license.
- PyTorch installation may need to be adjusted for local CUDA or CPU-only
  builds.
- CMA-ES workflows require `cma`.

## Experiments and Artifacts

`run_experiments.py` is intended to append dated CSV rows under
`artifacts/data/` for selected instance sizes and solvers:

```bash
python run_experiments.py --instance tree --solvers heuristic --start-users 10 --stop-users 30 --step-users 10
python run_experiments.py --instance parallel --solvers heuristic PPO --groups 4
```

The result schema is:

```text
instance,solver,users,neighbors,groups,score,runtime,buildtime
```

Generated and archived outputs are kept in `artifacts/`:

- `artifacts/data/`: CSV result tables.
- `artifacts/figures/`: generated plots and report figures.
- `artifacts/models/`: saved RL agents.

`plotting/result_plots.py` contains plotting helpers for instance solutions and
aggregate comparisons.

## Tests

Run the test suite with:

```bash
pytest
```

Some tests are marked as optional integration checks:

```bash
pytest -m "not integration"
pytest -m requires_pyomo
pytest -m requires_rl
pytest -m requires_cma
```

Current repository state note: `tests/` and `run_experiments.py` still import
legacy names such as `make_tree_instance`, `make_star_instance`, and
`instances.utility`, while the current `instances/builder.py` defines
`create_*_problem_instance` constructors. Those compatibility aliases or test
imports should be updated before treating the full test suite or experiment
runner as clean.

## Development Notes

- Keep data flow flat and NumPy-centric. The code is optimized for direct
  research iteration rather than framework-style abstraction.
- Prefer the compact solution representation
  `(placement, routing, quantities)` when implementing heuristics, RL policies,
  or evolutionary methods; convert to `(x_var, y_var, q_var)` only at solver
  boundaries.
- Generated files belong under `artifacts/`; reusable input topologies belong
  under `data/`.
- `archive/legacy/` is not part of the active API. Treat it as reference
  material.
- When adding a solver, expose a module-level `Solver` class and populate the
  `GenericSolver` output attributes after `run()`.

## Known Cleanup Targets

- Restore or replace legacy instance-builder aliases used by tests and
  `run_experiments.py`.
- Standardize naming between `x_var`/`x_variable`, `y_var`/`y_variable`, and
  `q_var`/`q_variable` in older modules.
- Add reproducibility metadata to experiment CSVs: commit hash, solver options,
  dependency versions, seed, and hardware notes.
- Move mathematical formulations and report notes into a dedicated `docs/`
  directory if the project grows beyond the current research-workbench layout.
