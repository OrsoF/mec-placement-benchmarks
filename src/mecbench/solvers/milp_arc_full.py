import numpy as np
import pyomo.environ as pyo

from mecbench.benchmarks.metrics import get_result_informations
from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.domain.problem import ProblemInstance
from mecbench.utils.arc_formulation import add_arc_constraints
from mecbench.utils.ilp_solvers import arc_model_arrays
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
        )  # x

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.UnitInterval
        )

        self.model.p = pyo.Var(self.model.total_number_of_demands, within=pyo.NonNegativeReals)

        self.model.P = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeReals,
        )

        self.model.L = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeReals,
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
            initialize={s: self.data.node_resources[s] for s in range(self.data.n_services)},
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
        x_variable, y_variable, q_variable = initial_value
        for demand_index in range(self.data.n_demands):
            self.model.p[demand_index].Start = q_variable[demand_index]
            for arc_index in range(self.data.n_arcs):
                self.model.demand_arc_transit[demand_index, arc_index].Start = x_variable[
                    demand_index, arc_index
                ]
        for service_index in range(self.data.n_services):
            for node_index in range(self.data.n_nodes):
                self.model.y[service_index, node_index].Start = y_variable[
                    service_index, node_index
                ]

    def run(
        self,
        thread_limit: int = 1,
        initial_value=None,
        options: dict | None = None,
    ):
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if initial_value is not None:
            self._set_initial_value(initial_value)

        opt = pyo.SolverFactory("gurobi")
        opt.options["threads"] = 1
        opt.options["Heuristics"] = 0
        opt.options.update(options or {})
        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)

        (
            self.runtime,
            self.lower_bound,
            self.upper_bound,
            self.solving_status,
        ) = get_result_informations(solver_result)

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
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
