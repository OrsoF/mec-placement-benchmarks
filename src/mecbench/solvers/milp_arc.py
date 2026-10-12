import pyomo.environ as pyo

from mecbench.benchmarks.metrics import get_result_informations
from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.domain.problem import ProblemInstance
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
        self.solver_name = "MILP_arc"

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.data.n_demands))
        self.model.S = pyo.Set(initialize=range(self.data.n_services))
        self.model.U = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.A = pyo.Set(initialize=range(self.data.n_arcs))

    def _build_variables(self):
        self.model.x = pyo.Var(self.model.D, self.model.A, within=pyo.Binary)
        self.model.y = pyo.Var(self.model.S, self.model.U, within=pyo.Binary)
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.P = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)
        self.model.L = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.D,
            initialize={d: self.data.origins[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.D,
            initialize={d: self.data.min_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.D,
            initialize={d: self.data.max_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.D,
            initialize={d: self.data.latency_limits[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.S,
            initialize={s: self.data.service_resources[s] for s in range(self.data.n_services)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.alpha = pyo.Param(
            self.model.A,
            initialize={edge: self.data.latency_slopes[edge] for edge in self.model.A},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.A,
            initialize={edge: self.data.latency_offsets[edge] for edge in self.model.A},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.serv_number = pyo.Param(
            self.model.D,
            initialize={d: int(self.data.demand_to_service[d]) for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.out_arcs = pyo.Param(
            self.model.U,
            initialize={
                u: get_encoded_outgoing_arc(self.graph, self.data.arc_index, u)
                for u in range(self.data.n_nodes)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.U,
            initialize={
                u: get_encoded_incoming_arc(self.graph, self.data.arc_index, u)
                for u in range(self.data.n_nodes)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.ind_u_origin = pyo.Param(
            self.model.U,
            self.model.D,
            initialize={
                (u, d): int(u == self.model.origins[d])
                for u in range(self.data.n_nodes)
                for d in range(self.data.n_demands)
            },
            mutable=False,
            within=pyo.Binary,
        )

        self.model.node_resources = pyo.Param(
            self.model.U,
            initialize={u: self.data.node_resources[u] for u in range(self.data.n_nodes)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={a: self.data.arc_capacities[a] for a in range(self.data.n_arcs)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.N = pyo.Param(
            initialize=max(self.data.latency_limits), mutable=False, within=pyo.NonNegativeReals
        )
        self.model.M = pyo.Param(
            initialize=max(self.data.max_flows), mutable=False, within=pyo.NonNegativeIntegers
        )

    def _build_constraints(self):
        add_arc_constraints(
            self.model,
            self.model.D,
            self.model.A,
            self.model.U,
            self.model.S,
            self.model.x,
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def run(
        self,
        options: dict = None,
        solving_time_limit: float = 600,
        thread_limit: int = 1,
        solver: str = "gurobi",
    ):
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if solver == "gurobi":
            opt = configure_gurobi(
                pyo.SolverFactory(solver),
                options,
                time_limit=solving_time_limit,
                thread_limit=thread_limit,
            )
        else:
            raise NotImplementedError("Only Gurobi is currently supported.")

        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)
        (self.runtime, self.lower_bound, self.upper_bound, self.solving_status) = (
            get_result_informations(solver_result)
        )
        self._set_output_variables(
            *arc_model_arrays(
                self.model.x,
                self.model.y,
                self.model.p,
                self.data.n_demands,
                self.data.n_arcs,
                self.data.n_services,
                self.data.n_nodes,
            )
        )
        self.score = objective_value(self.q_var)
        self.build_time = 0.0
