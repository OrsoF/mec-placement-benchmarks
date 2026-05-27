# Edge Computing Service Placement Experiments

This repository contains a research/prototyping codebase for service placement and
routing in edge-computing networks. It builds graph-based problem instances,
solves them with exact mathematical-programming models, heuristics, and
reinforcement-learning policies, then stores experiment outputs for comparison.

The central optimization problem is:

- place each service on one network node with enough compute resources;
- route each demand from its origin to the node hosting the requested service;
- choose served flow quantities within demand bounds;
- respect edge capacities and latency constraints;
- maximize the total served quantity.

## Repository Layout

```text
.
|-- main.py                 # Batch runner for tree and parallel/star experiments
|-- requirements.txt        # Python dependencies
|-- main.tex                # LaTeX table snippet for reported results
|-- core/                   # Domain model, graph helpers, placement helpers, solver contract
|-- solvers/                # Exact, relaxation, heuristic, evolutionary, and RL solvers
|-- instances/              # Synthetic and GML-based instance builders
|-- rl/                     # Gym environment, RL policies, and RL solver base class
|-- data_io/                # Pickle, result parsing, and artifact I/O helpers
|-- plotting/               # Plot generation helpers
|-- data/topologies/        # Input network topologies in GML format
|-- experiments/            # Exploratory experiment entry points
|-- artifacts/              # Generated CSVs, plots, PDFs, pickles, and trained agents
|-- archive/legacy/         # Legacy scripts and notebooks kept for reference
`-- tests/                  # Unit tests and smoke checks
```

## Main Concepts

### Problem Instances

`core/problem_classes/` defines the objects shared by all solvers:

- `NetworkGraph`: wraps a NetworkX graph with node resources, edge capacities,
  and latency parameters.
- `Demand`: describes an origin node, minimum/maximum flow, and maximum latency.
- `Service`: groups one or more demands and a compute-resource requirement.
- `ServicePackage`: groups services for one experiment instance.
- `ProblemInstance`: combines the network and services, validates basic
  feasibility, and exposes vector/matrix data used by solvers.
- `SolvingInstance`: provides solution validation and helper operations such as
  shortest-path routing and greedy quantity maximization.

### Instance Builders

`instances/` contains generators for several benchmark families:

- `build_tree.py`: tree-like edge/cloud topology with one service per
  user.
- `build_parallel.py`: grouped star/parallel topology.
- `build_gml.py`: instances derived from GML topologies in
  `data/topologies/`.
- `build_large.py`: larger synthetic instances.

### Solvers

`solvers/` contains several approaches:

- `solvers/exact/`: arc-based and path-based MILP/ILP formulations with Pyomo
  and Gurobi, including warm-started variants.
- `solvers/relaxations/`: LP relaxation formulations.
- `solvers/heuristic/`: local-search heuristics.
- `solvers/evolutionary/`: CMA-ES experiments.
- `solvers/rl/`: PPO/A2C reinforcement-learning solvers using
  Stable-Baselines3 and Gymnasium.

The common interface is `core/generic_solver.py`: solvers receive a
`ProblemInstance`, run, and expose `score`, `runtime`, and solution variables.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Some solvers require additional system setup:

- Gurobi must be installed and licensed for the Pyomo MILP/LP solvers.
- RL solvers depend on Stable-Baselines3, Gymnasium, PyTorch, and
  Torch-Geometric. The versions may need to match your local CUDA/CPU setup.

## Running Experiments

Run the main batch experiment on a tree topology:

```bash
python main.py --instance tree
```

Run the grouped parallel/star topology experiment:

```bash
python main.py --instance parallel
```

The script writes a date-stamped CSV file such as `results_26_05_26.csv` under
`artifacts/data/`. Existing plotted outputs and trained agents are kept under
`artifacts/`.

## Development Notes

- The repository is currently organized as a research workbench. Some scripts in
  `experiments/` and `archive/legacy/` are exploratory and may require manual
  adjustment before reuse.
- The test suite in `tests/` covers core domain models, instance builders, Gym
  environment behavior, heuristic behavior, and solver contracts.
- Several modules use fixed random seeds, which is useful for repeatability but
  should be reviewed before large experimental campaigns.
- Generated artifacts are organized under `artifacts/`. If results become large
  or are regenerated often, consider moving them to an artifact store or a
  separate results repository.

## Suggested Next Improvements

1. Move experiment configuration out of `main.py` into YAML/JSON or CLI flags.
2. Standardize solver output names (`x_var`/`x_variable`, `y_var`/`y_variable`,
   `q_var`/`q_variable`) to reduce adapter code.
3. Add reproducibility metadata to result CSVs: commit hash, solver options,
   dependency versions, seed, and hardware notes.
4. Add a `docs/` folder for mathematical formulation notes and report sources.

A cleaner long-term layout could look like this:

```text
.
|-- core/
|-- solvers/
|-- instances/
|-- rl/
|-- data_io/
|-- plotting/
|-- experiments/
|-- configs/
|-- data/
|-- artifacts/
|-- archive/
|-- tests/
|-- docs/
|-- main.py
|-- requirements.txt
`-- README.md
```

This keeps the repository as one practical research workspace while making the
purpose of each folder clearer.
