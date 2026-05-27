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
|-- core/                   # Core domain model: graph, services, demands, instances
|-- solvers/                # MILP/LP, heuristic, CMA-ES, and RL solver implementations
|-- utils/                  # Instance builders, Gym environments, plotting, I/O helpers
|-- gml_graphs/             # Input network topologies in GML format
|-- exps/                   # Exploratory experiment scripts
|-- artifact/               # Organized generated CSVs, plots, PDFs, and trained RL artifacts
|-- results/                # Empty staging area for newly generated outputs
`-- old/                    # Legacy scripts and notebooks kept for reference
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

`utils/` contains generators for several benchmark families:

- `build_instance_tree.py`: tree-like edge/cloud topology with one service per
  user.
- `build_instance_parallel.py`: grouped star/parallel topology.
- `build_instance_gml.py`: instances derived from GML topologies in
  `gml_graphs/`.
- `build_instance_large.py`: larger synthetic instances.

### Solvers

`solvers/` contains several approaches:

- arc-based and path-based MILP/LP formulations with Pyomo and Gurobi;
- warm-started MILP variants using heuristic or RL solutions;
- local-search heuristics;
- CMA-ES experiments;
- PPO/A2C reinforcement-learning solvers using Stable-Baselines3 and Gymnasium.

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

The script writes a date-stamped CSV file such as `results_26_05_26.csv` in the
current working directory. Existing plotted outputs and trained agents are kept
under `artifact/`.

## Development Notes

- The repository is currently organized as a research workbench. Some scripts in
  `exps/` and `old/` are exploratory and may require manual adjustment before
  reuse.
- The test suite in `tests/` covers core domain models, instance builders, Gym
  environment behavior, heuristic behavior, and solver contracts.
- Several modules use fixed random seeds, which is useful for repeatability but
  should be reviewed before large experimental campaigns.
- Generated artifacts are organized under `artifact/`. If results become large
  or are regenerated often, consider moving them to an artifact store or a
  separate results repository.

## Suggested Next Improvements

1. Add a small test suite around instance construction and solution validation.
2. Move experiment configuration out of `main.py` into YAML/JSON or CLI flags.
3. Standardize solver output names (`x_var`/`x_variable`, `y_var`/`y_variable`,
   `q_var`/`q_variable`) to reduce adapter code.
4. Move solver contracts and graph path helpers into `core/` if the project
   keeps tightening the domain boundary.
5. Split the broad `utils/` directory into more explicit top-level folders:
   `instances/`, `rl/`, `plotting/`, and `io/`.
6. Separate active experiments from archived scripts, for example by replacing
   `old/` with tagged releases or a clearly documented archive.
7. Add reproducibility metadata to result CSVs: commit hash, solver options,
   dependency versions, seed, and hardware notes.

A cleaner long-term layout could look like this:

```text
.
|-- core/                  # Domain model: graph, services, demands, instances
|-- solvers/               # Exact, heuristic, CMA-ES, and RL solver classes
|-- instances/             # Synthetic and GML-based instance builders
|-- rl/                    # Gym environments, policies, and RL-specific helpers
|-- plotting/              # Plot generation and result visualization
|-- experiments/           # Reproducible experiment entry points
|-- configs/               # Experiment parameter files
|-- data/                  # Input topologies and reusable datasets
|-- results/               # Generated CSVs, plots, trained agents, and reports
|-- tests/                 # Unit tests and regression checks
|-- docs/                  # Mathematical formulation and research notes
|-- main.py                # Lightweight CLI entry point
|-- requirements.txt
`-- README.md
```

This keeps the repository as one practical research workspace while making the
purpose of each folder clearer.
