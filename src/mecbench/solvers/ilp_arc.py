import time

import numpy as np
import pyomo.environ as pyo

from mecbench.benchmarks.metrics import get_result_informations
from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.heuristic import Solver as Heursolv
from mecbench.utils.arc_formulation import add_arc_constraints
from mecbench.utils.ilp_solvers import (
    arc_model_arrays,
    configure_gurobi,
    objective_value,
)
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "ILP_arc"

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.total_number_of_demands = pyo.Set(initialize=range(self.data.n_demands))
        self.model.number_of_services = pyo.Set(initialize=range(self.data.n_services))
        self.model.node_number = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.arc_number = pyo.Set(initialize=range(self.data.n_arcs))

    def _build_variables(self):
        self.model.demand_arc_transit = pyo.Var(
            self.model.total_number_of_demands, self.model.arc_number, within=pyo.Binary
        )

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.Binary
        )

        self.model.p = pyo.Var(self.model.total_number_of_demands, within=pyo.NonNegativeIntegers)

        self.model.P = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

        self.model.L = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.total_number_of_demands,
            initialize={d: self.data.origins[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.total_number_of_demands,
            initialize={d: self.data.min_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.total_number_of_demands,
            initialize={d: self.data.max_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.total_number_of_demands,
            initialize={d: self.data.latency_limits[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.number_of_services,
            initialize={s: self.data.service_resources[s] for s in range(self.data.n_services)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.alpha = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.data.latency_slopes[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.data.latency_offsets[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.serv_number = pyo.Param(
            self.model.total_number_of_demands,
            initialize={d: int(self.data.demand_to_service[d]) for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.out_arcs = pyo.Param(
            self.model.node_number,
            initialize={
                u: get_encoded_outgoing_arc(self.graph, self.data.arc_index, u)
                for u in range(self.data.n_nodes)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.node_number,
            initialize={
                u: get_encoded_incoming_arc(self.graph, self.data.arc_index, u)
                for u in range(self.data.n_nodes)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.ind_u_origin = pyo.Param(
            self.model.node_number,
            self.model.total_number_of_demands,
            initialize={
                (u, d): int(u == self.model.origins[d])
                for u in range(self.data.n_nodes)
                for d in range(self.data.n_demands)
            },
            mutable=False,
            within=pyo.Binary,
        )

        self.model.node_resources = pyo.Param(
            self.model.node_number,
            initialize={u: self.data.node_resources[u] for u in range(self.data.n_nodes)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.arc_capacity = pyo.Param(
            self.model.arc_number,
            initialize={a: self.data.arc_capacities[a] for a in range(self.data.n_arcs)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.N = pyo.Param(
            initialize=max(self.data.latency_limits), mutable=False, within=pyo.NonNegativeReals
        )

        self.model.M = pyo.Param(
            initialize=max(self.data.max_flows),
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

    def _build_constraints(self):
        add_arc_constraints(
            self.model,
            self.model.total_number_of_demands,
            self.model.arc_number,
            self.model.node_number,
            self.model.number_of_services,
            self.model.demand_arc_transit,
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def _set_initial_value(self, initial_value: tuple):
        x_var, y_var, q_var = initial_value
        for demand_index in range(self.data.n_demands):
            self.model.p[demand_index].value = q_var[demand_index]
            for arc_index in range(self.data.n_arcs):
                self.model.demand_arc_transit[demand_index, arc_index].value = x_var[
                    demand_index, arc_index
                ]
        for service_index in range(self.data.n_services):
            for node_index in range(self.data.n_nodes):
                self.model.y[service_index, node_index].value = y_var[service_index, node_index]

    def set_heuristic_placement(self):
        cloud_server = np.argmax(self.data.node_resources)

        print(cloud_server)

        for service in range(self.data.n_services):
            self.model.y[service, cloud_server].fix(1)

    def set_heuristic_solution(self, seed=0):
        heuristic_solver = Heursolv(self.problem_instance, seed=seed)
        heuristic_solver.run()
        self._set_initial_value(
            (heuristic_solver.x_var, heuristic_solver.y_var, heuristic_solver.q_var)
        )

    def set_rl_solution(self, seed=0):
        from mecbench.solvers.ppo import Solver as RlPPO

        solver = RlPPO(self.problem_instance, seed=seed)
        solver.run()

        best_state = solver.env.best_state_record[0]

        x_var, y_var, q_var = solver.env._state_to_variables(best_state)
        self._set_initial_value((x_var, y_var, q_var))

    def run(
        self,
        initial_value: tuple = None,
        options: dict | None = None,
        heuristic_placement: bool = False,
        heuristic_solution: bool = False,
        rl_solution: bool = False,
    ):
        start_time = time.time()

        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if initial_value is not None:
            self._set_initial_value(initial_value)

        if heuristic_placement:
            self.set_heuristic_placement()

        if heuristic_solution:
            self.set_heuristic_solution(seed=(options or {}).get("Seed", 0))

        if rl_solution:
            self.set_rl_solution(seed=(options or {}).get("Seed", 0))

        opt = configure_gurobi(
            pyo.SolverFactory("gurobi"),
            {"Heuristics": 0, "MemLimit": 8, **(options or {})},
        )

        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)
        solver_runtime = get_result_informations(solver_result)[0]

        self._set_output_variables(
            *arc_model_arrays(
                self.model.demand_arc_transit,
                self.model.y,
                self.model.p,
                self.data.n_demands,
                self.data.n_arcs,
                self.data.n_services,
                self.data.n_nodes,
            )
        )

        try:
            self.score = objective_value(self.q_var)
        except (TypeError, ValueError):
            self.score = 0

        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
