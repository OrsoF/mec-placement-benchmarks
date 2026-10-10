from mecbench.solvers.base import GenericSolver
from mecbench.solvers.model_utils import configure_gurobi, objective_value, problem_arrays
from mecbench.domain.problem import Service
from mecbench.domain.problem import ProblemInstance
from mecbench.benchmarks.metrics import get_result_informations
from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.solvers.arc_formulation import add_arc_constraints

import pyomo.environ as pyo
import time
import networkx as nx
import numpy as np


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "MILP_arc"

    def _model_parameters(self):
        a = problem_arrays(self.problem_instance)
        self.S, self.D, self.U, self.A = a.S, a.D, a.U, a.A
        self.Sd = self.problem_instance.demand_per_service
        self.arc_dic = a.arc_index
        self.arc_list = a.arc_list
        self.origins = a.origins
        self.Pdmin = a.qmin
        self.Pdmax = a.qmax
        self.latencies = a.latencies
        self.resources = a.resources
        self.alpha = a.alpha
        self.beta = a.beta

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.D))
        self.model.S = pyo.Set(initialize=range(self.S))
        self.model.U = pyo.Set(initialize=range(self.U))
        self.model.A = pyo.Set(initialize=range(self.A))

    def _build_variables(self):
        self.model.x = pyo.Var(self.model.D, self.model.A, within=pyo.Binary)
        self.model.y = pyo.Var(self.model.S, self.model.U, within=pyo.Binary)
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.P = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)
        self.model.L = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.D,
            initialize={d: self.origins[d] for d in range(self.D)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.D,
            initialize={d: self.Pdmin[d] for d in range(self.D)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.D,
            initialize={d: self.Pdmax[d] for d in range(self.D)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.D,
            initialize={d: self.latencies[d] for d in range(self.D)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.S,
            initialize={s: self.resources[s] for s in range(self.S)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.alpha = pyo.Param(
            self.model.A,
            initialize={edge: self.alpha[edge] for edge in self.model.A},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.A,
            initialize={edge: self.beta[edge] for edge in self.model.A},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.serv_number = pyo.Param(
            self.model.D,
            initialize={
                sum(len(elem.task_list) for elem in self.services[:i]) + j: i
                for i, service in enumerate(self.services)
                for j, task in enumerate(service.task_list)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.out_arcs = pyo.Param(
            self.model.U,
            initialize={
                u: get_encoded_outgoing_arc(self.G, self.arc_dic, u)
                for u in range(self.U)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.U,
            initialize={
                u: get_encoded_incoming_arc(self.G, self.arc_dic, u)
                for u in range(self.U)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.ind_u_origin = pyo.Param(
            self.model.U,
            self.model.D,
            initialize={
                (u, d): int(u == self.model.origins[d])
                for u in range(self.U)
                for d in range(self.D)
            },
            mutable=False,
            within=pyo.Binary,
        )

        self.model.node_resources = pyo.Param(
            self.model.U,
            initialize={
                u: self.G.nodes[u]["available_resources"] for u in range(self.U)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={
                a: self.G.edges[edge]["capacity"] for a, edge in enumerate(self.G.edges)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.N = pyo.Param(
            initialize=max(self.latencies), mutable=False, within=pyo.NonNegativeReals
        )
        self.model.M = pyo.Param(
            initialize=max(self.Pdmax), mutable=False, within=pyo.NonNegativeIntegers
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
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.p.values())
        )

    def run(
        self,
        options: dict = None,
        solving_time_limit: float = 600,
        thread_limit: int = 1,
        solver: str = "gurobi",
    ):
        self._model_parameters()
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
            assert False, "Not implemented yet."

        solver_result = opt.solve(self.model)
        self._record_solver_result(solver_result)
        (self.runtime, self.lower_bound, self.upper_bound, self.solving_status) = get_result_informations(solver_result)
        self._set_arc_model_solution(
            self.model.x, self.model.y, self.model.p, self.D, self.A, self.S, self.U,
        )
        self.score = objective_value(self.q_var)
        self.build_time = 0.0
