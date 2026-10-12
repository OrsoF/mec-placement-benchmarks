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