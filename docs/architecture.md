# Architecture and migration

`mecbench` is installed from `src/`. Runtime imports use that namespace;
there is no need to add the repository root to `sys.path`.

| Previous module | Current module |
| --- | --- |
| `core.problem_classes.*` | `mecbench.domain.problem` |
| `core.solution`, `core.problem_classes.solving_instance` | `mecbench.domain.solution` |
| `core.feasibility` | `mecbench.domain.validation` |
| `core.generic_solver` | `mecbench.solvers.base` |
| `core.pyomo_model` | `mecbench.solvers.model_utils` |
| `instances.builder` | `mecbench.instances.synthetic` |
| `instances.graph_build` | `mecbench.instances.topology` |
| `solvers.heuristic` | `mecbench.solvers.heuristics` |
| `solvers.exact`, `solvers.relaxations` | `mecbench.solvers.exact`, `mecbench.solvers.relaxations` |
| `solvers.evolutionary`, `solvers.rl` | `mecbench.solvers.evolutionary`, `mecbench.solvers.rl` |
| `rl.*` | `mecbench.environments.*` |
| `data_io.experiment_io`, `mecbench.benchmarks.io` | `mecbench.compat.experiment_io` |
| `mecbench.benchmarks.pickle_io` | `mecbench.compat.pickle_io` |
| `plotting.result_plots` | `mecbench.benchmarks.reporting` |
| `exp.py` | `mecbench.cli`, `mecbench.benchmarks.runner` |

The domain defines demands, networks, services, packages, and problems.
`domain.network` owns graph data, `domain.services` owns demands and services,
and `domain.problem` assembles a problem. `domain.solution.Solution` owns copied,
read-only arrays. Pure compact/arc conversion lives in `domain.conversions`;
Load, residual and total-traffic latency calculations live in `domain.loads`
and are shared by validation, reporting and graph updates.

`solvers.workspace.SolverWorkspace` owns a mutable graph, candidate state, path
cache, and independent random generator. Candidate search lives in `solvers.search`,
graph updates in `solvers.graph_state`, baseline construction in
`solvers.initialization`, and indexed optimizer extraction in
`solvers.exact.extraction`. Workspaces store their results without changing the
problem's historical solution state. RL environments use the workspace's copy
of the directed graph and its path cache, so observations never mutate the input
problem or another environment. `GenericSolver` initializes outputs through
an ordinary constructor; subclasses explicitly call `super().__init__`.

All feasibility decisions use `domain.validation`; its exception is
`SolutionValidationError`, a `ValueError`. Flow conservation and shared arc
latency follow the arc MILP, including its allowance of balanced circulations.

Historical imports from `domain.problem` and `domain.solution` remain available
as reexports. `SolvingInstance` aliases `SolverWorkspace`. Historical mutable
`ProblemInstance.x_variable`, `y_variable`, `q_variable`, `_set_arc_solution`, and
`validate_instance` live in `compat.problem_state`; they initialize a baseline
only on first access. `validate_instance` checks that particular baseline and
does not prove that the problem is infeasible. New code should use `solve` and
`validate`, and import helpers from their canonical modules.

The public facade in `api.py` exports instance builders and `solve`. Results
carry a copied `Solution`, a `ValidationReport`, termination status and objective
recomputed from validated quantities. `solvers.registry` provides the shared
lazy registry, including historical aliases. The CLI and benchmark runner use
this facade. See [the API guide](api.md) for the public contracts.

Instance construction, environments, and solvers depend on the domain.
Benchmark orchestration depends on those components, validates solver outputs,
and records results through `metrics`. Reporting is imported only when plots
are requested. CLI imports do not require Pyomo, Gurobi, or RL packages.

Tests are organized by purpose: constraint tests in `unit`, component and CLI
tests in `integration`, and shared-load counterexamples plus comparison with
actual Pyomo MILP constraints in `regression`. The CI runs minimal and optional
dependency profiles on Windows and Linux with Python 3.10 and 3.12, runs a
separate RL profile, builds distributions, and checks Docker. Ruff covers active
Python sources, tests, and examples; strict mypy coverage grows from an explicit
list of public contracts and adapters. See [development](development.md).

`examples/` contains the quickstart, solver smoke runner and environment rendering
example. The former `exp.py`, `exp_tree.py`, `exp_star.py`, and `reset.py` wrappers
were removed; use the CLI for tree/star experiments. The historical experiment
API and its single pickle implementation live in `compat/`. Archived scripts
and notebooks reference those modules; the archived allocation helper lives
beside its only caller in `archive/legacy/service_placement.py`.

Unused `instances.utility`, `ToyInstance`, GML/path reconstruction helpers,
old graph-update functions and unused plotting/reward aliases were removed.
`data/`, `artifacts/`,
and `archive/` retain input topologies, experiment outputs, and historical work.
Archived imports were updated; historical algorithms and saved notebook outputs
are retained. Old module namespaces are removed. Previously pickled Python
objects that reference those namespaces must be regenerated or migrated before
loading; filenames alone cannot migrate pickle class references.

File names are lowercase except the standard project files `README.md`,
`LICENSE`, `Dockerfile`, and `MANIFEST.in`. The setuptools manifest includes
tests, examples, documentation, and topology files in source distributions.
