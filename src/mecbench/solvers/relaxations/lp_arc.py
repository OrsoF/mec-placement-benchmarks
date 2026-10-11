import time

import networkx as nx
import numpy as np
import pyomo.environ as pyo

from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.services import Service
from mecbench.solvers.arc_formulation import add_arc_constraints
from mecbench.solvers.base import GenericSolver
from mecbench.solvers.exact.extraction import arc_model_arrays


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "LP_arc"

    def _model_parameters(self):
        self.S = len(self.services)
        self.Sd = [service.task_number for service in self.services]
        self.D = sum(self.Sd)
        self.U = self.G.number_of_nodes()
        self.A = self.G.number_of_edges()
        self.arc_dic = {elem: i for i, elem in enumerate(self.G.edges)}
        self.arc_list = list(self.G.edges)
        self.origins = [origin for service in self.services for origin in service.origin_nodes]
        self.Pdmin = [min_flow for service in self.services for min_flow in service.min_flows]
        self.Pdmax = [max_flow for service in self.services for max_flow in service.max_flows]
        self.latencies = [latency for service in self.services for latency in service.max_latencies]
        self.resources = [
            service.needed_resources for service in self.services for _ in service.task_list
        ]
        self.alpha = [self.G.edges[edge]["alpha"] for edge in self.G.edges]
        self.beta = [self.G.edges[edge]["beta"] for edge in self.G.edges]

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.D))
        self.model.S = pyo.Set(initialize=range(self.S))
        self.model.U = pyo.Set(initialize=range(self.U))
        self.model.A = pyo.Set(initialize=range(self.A))

    def _build_variables(self):
        self.model.x = pyo.Var(
            self.model.D, self.model.A, within=pyo.NonNegativeReals, bounds=(0, 1)
        )

        self.model.y = pyo.Var(
            self.model.S, self.model.U, within=pyo.NonNegativeReals, bounds=(0, 1)
        )
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
            initialize={edge: self.alpha[edge] for edge in self.model.A},
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.A,
            initialize={edge: self.beta[edge] for edge in self.model.A},
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
                u: get_encoded_outgoing_arc(self.G, self.arc_dic, u) for u in range(self.U)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.U,
            initialize={
                u: get_encoded_incoming_arc(self.G, self.arc_dic, u) for u in range(self.U)
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
            initialize={u: self.G.nodes[u]["available_resources"] for u in range(self.U)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={a: self.G.edges[edge]["capacity"] for a, edge in enumerate(self.G.edges)},
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
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def _build_whole_problem(self):
        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

    def run(
        self,
        options: dict | None = None,
    ):
        start_time = time.time()

        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        opt = pyo.SolverFactory("gurobi")
        opt.options.update(options or {})
        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)

        self._set_output_variables(
            *arc_model_arrays(
                self.model.x,
                self.model.y,
                self.model.p,
                self.D,
                self.A,
                self.S,
                self.U,
            )
        )
        self.runtime = time.time() - start_time
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
