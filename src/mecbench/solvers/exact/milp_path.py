from mecbench.solvers.base import GenericSolver
from mecbench.domain.problem import Service
from mecbench.domain.problem import ProblemInstance
from mecbench.benchmarks.metrics import get_result_informations
from mecbench.solvers.path_formulation import (
    add_path_constraints,
    add_path_params,
    enumerate_candidate_paths,
)

import pyomo.environ as pyo
from time import time
import networkx as nx
import numpy as np


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, maximum_path_number: int = 1):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.limit = maximum_path_number
        self.solver_name = "MILP_path"

    def _model_parameters(self):
        self.S = len(self.services)
        self.Sd = [service.task_number for service in self.services]
        self.D = sum(self.Sd)
        self.U = self.G.number_of_nodes()
        self.A = self.G.number_of_edges()
        self.arc_dic = {elem: i for i, elem in enumerate(self.G.edges)}
        self.arc_list = list(self.G.edges)
        self.origins = [
            origin for service in self.services for origin in service.origin_nodes
        ]
        self.Pdmin = [
            min_flow for service in self.services for min_flow in service.min_flows
        ]
        self.Pdmax = [
            max_flow for service in self.services for max_flow in service.max_flows
        ]
        self.latencies = [
            latency for service in self.services for latency in service.max_latencies
        ]
        self.resources = [
            service.needed_resources
            for service in self.services
            for _ in service.task_list
        ]
        self.alpha = [self.G.edges[edge]["alpha"] for edge in self.G.edges]
        self.beta = [self.G.edges[edge]["beta"] for edge in self.G.edges]
        (
            self.shortest_paths,
            self.all_shortest_paths,
            self.D_path_number,
            self.path_start,
        ) = enumerate_candidate_paths(self.G, self.origins, self.limit)
        self.m = max(self.Pdmax)
        self.n = max(self.latencies)

        self.C = len(self.all_shortest_paths)
        self.D_path = self.D_path_number

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.D))
        self.model.U = pyo.Set(initialize=range(self.U))
        self.model.S = pyo.Set(initialize=range(self.S))
        self.model.A = pyo.Set(initialize=range(self.A))
        self.model.C = pyo.Set(initialize=range(self.C))

    def _build_variables(self):
        self.model.l = pyo.Var(self.model.C, within=pyo.Binary)
        self.model.y = pyo.Var(self.model.S, self.model.U, within=pyo.Binary)
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.P = pyo.Var(self.model.C, within=pyo.NonNegativeReals)
        self.model.L = pyo.Var(self.model.C, self.model.A, within=pyo.NonNegativeReals)
        self.model.nul = pyo.Var(range(1), within=pyo.Binary)
        self.model.nul[0].fix(0)

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
        self.model.serv_number = pyo.Param(
            self.model.D,
            initialize={
                sum(service.task_number for service in self.services[:i]) + j: i
                for i, service in enumerate(self.services)
                for j, task in enumerate(service.task_list)
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

        self.model.node_resources = pyo.Param(
            self.model.U,
            initialize={
                u: self.G.nodes[u]["available_resources"] for u in range(self.U)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        add_path_params(
            self.model,
            self.shortest_paths,
            self.all_shortest_paths,
            self.D_path_number,
            self.path_start,
            self.arc_dic,
            self.arc_list,
            self.Pdmax,
            self.latencies,
        )

    def _build_constraints(self):
        add_path_constraints(self.model, self.m, self.n)

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

        opt = pyo.SolverFactory(solver)
        if solver == "gurobi":
            opt.options["TimeLimit"] = solving_time_limit
            opt.options["threads"] = thread_limit
        else:
            assert False, "Not implemented yet."
        opt.options.update(options or {})

        solver_result = opt.solve(self.model)
        self._record_solver_result(solver_result)
        (self.runtime, self.lower_bound, self.upper_bound, self.solving_status) = get_result_informations(solver_result)
        self._set_path_model_solution(
            self.model.l, self.model.y, self.model.p, self.all_shortest_paths,
            self.D_path_number, self.arc_dic, self.D, self.A, self.S, self.U,
        )
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
