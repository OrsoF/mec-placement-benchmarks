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
| `data_io.experiment_io` | `mecbench.benchmarks.io` |
| `plotting.result_plots` | `mecbench.benchmarks.reporting` |
| `exp.py` | `mecbench.cli`, `mecbench.benchmarks.runner` |

The domain defines demands, networks, services, packages, and problems.
`domain.solution` handles compact/arc conversion and the `SolvingInstance`
adapter. All feasibility decisions use `domain.validation`; its exception is
`SolutionValidationError`, a `ValueError`. Flow conservation and shared arc
latency follow the arc MILP, including its allowance of balanced circulations.

Instance construction, environments, and solvers depend on the domain.
Benchmark orchestration depends on those components, validates solver outputs,
and records results through `metrics`. Reporting is imported only when plots
are requested. CLI imports do not require Pyomo, Gurobi, or RL packages.

Tests are organized by purpose: constraint tests in `unit`, component and CLI
tests in `integration`, and shared-load counterexamples plus comparison with
actual Pyomo MILP constraints in `regression`. The CI runs the suite on Windows
and Linux with Python 3.10 and 3.12, builds distributions, and checks Docker.

`examples/` contains the previous executable scripts. `data/`, `artifacts/`,
and `archive/` retain input topologies, experiment outputs, and historical work.
Archived imports were updated; historical algorithms and saved notebook outputs
are retained. Old module namespaces are removed. Previously pickled Python
objects that reference those namespaces must be regenerated or migrated before
loading; filenames alone cannot migrate pickle class references.

File names are lowercase except the standard project files `README.md`,
`LICENSE`, `Dockerfile`, and `MANIFEST.in`. The setuptools manifest includes
tests, examples, documentation, and topology files in source distributions.
