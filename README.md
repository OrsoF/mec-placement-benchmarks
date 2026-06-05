# Edge Computing Service Placement

Research code for the service placement problem in edge-computing networks.
This repository contains the code used to produce the experimental graphs for
the IFIP Networking article `ifip_forghieri.pdf`, "Faster Latency Constrained
Service Placement in Edge Computing with Deep Reinforcement Learning". The
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
|-- exp_tree.py             # Tree benchmark driver and plots
|-- exp_star.py             # Star benchmark driver and plots
|-- ifip_forghieri.pdf      # Article associated with the generated graphs
|-- requirements.txt        # Python dependency list
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

## Article Experiments and Artifacts

The top-level experiment scripts append CSV rows under `artifacts/data/` and
generate the objective/runtime figures used for article-style comparisons.
They benchmark the exact ILP arc formulation, the use-case heuristic, and PPO
on increasing graph sizes.

```bash
python exp_tree.py --solvers ilp_arc heuristic PPO --start-users 20 --stop-users 410 --step-users 10
python exp_star.py --solvers ilp_arc heuristic PPO --groups 4 --start-users 15 --stop-users 125 --step-users 5
```

The result schema is:

```text
instance,solver,users,neighbors,groups,score,runtime,buildtime
```

Each script now produces two figures from its output CSV:

- `artifacts/figures/tree/tree_objective_vs_graph_size.pdf`
- `artifacts/figures/tree/tree_runtime_vs_graph_size.pdf`
- `artifacts/figures/star/star_objective_vs_graph_size.pdf`
- `artifacts/figures/star/star_runtime_vs_graph_size.pdf`

Use `--figures-dir` to redirect these plots and `--no-plots` to only append
CSV rows. Failed solver runs are kept in the CSV with blank objective values;
the runtime plot still includes their elapsed time.

Generated and archived outputs are kept in `artifacts/`:

- `artifacts/data/`: CSV result tables.
- `artifacts/figures/`: generated plots and report figures.
- `artifacts/models/`: saved RL agents.

`plotting/result_plots.py` contains plotting helpers for instance solutions and
aggregate comparisons.

The article formulates the Service Placement Problem as joint service placement,
flow selection, and routing under node-resource, edge-capacity, and latency
constraints. It compares a linearized ILP, a structure-aware heuristic, and deep
RL. The reported conclusion is that exact ILP solving does not scale cleanly to
larger or dynamic orchestration instances, while RL can reuse pre-training and
adapt quickly, with lower solution quality than the optimum but much shorter
online runtime.

## Hierarchical RL Directions

The current RL environment exposes the full orchestration action as a single
decision process. A stronger hierarchical approach should use the natural
three-part structure already present in the heuristic and formulation:

- High-level placement policy: choose service-to-node placements using graph,
  residual resources, demand origins, and cloud/edge/user layer features. This
  policy acts less frequently and defines the feasible subproblem for lower
  levels.
- Routing option policy: given a placement, choose candidate paths or next-hop
  options for each demand from a cached k-shortest-path or latency-feasible path
  set. This reduces the action space relative to per-edge routing decisions.
- Flow-control policy: assign or adjust `q` on the selected routes using
  residual edge capacity and latency slack. This can be trained as a fast local
  repair/improvement policy around a feasible baseline.
- Feasibility-first options: make each subpolicy terminate with a feasible
  partial solution whenever possible, using the cloud placement and minimum QoS
  as the fallback option. Penalize only real constraint violations after local
  repair, rather than making the top-level policy learn all feasibility details.
- Warm-start hybrid solving: use the hierarchical policy to generate
  `(placement, routing, quantities)` warm starts for `ilp_arc` or the precise
  heuristic, then let the exact/local solver repair or certify the solution
  within a short time budget.
- Transfer across graph sizes: pre-train placement/routing options on small
  tree and star instances, freeze or slowly update the option encoders, and
  fine-tune only the high-level scheduler on larger or dynamic instances.
- Graph-aware observations: replace flat observations with node/edge embeddings
  from a small message-passing network or handcrafted aggregate features
  grouped by user, WAP/edge, and cloud layers. Keep the action heads tied to
  feasible node/path sets to avoid dimension growth from invalid actions.

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
