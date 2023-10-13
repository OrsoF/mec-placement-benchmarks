from utils.instance_solver import InstanceSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc
from itertools import islice
from utils.data_management import get_result_informations
from pyomo.opt.base.solvers import SolverFactoryClass

import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os


class Solver:
    def __init__(
        self,
        problem_instance: ProblemInstance,
        maximum_path_number: int = 1,
        eps: float = 0.0001,
    ):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.limit = maximum_path_number
        self.solver_name = "LP_path_strong_latency"
        self.eps = eps

    def _model_parameters(self):
        self.S = len(self.services)
        self.number_of_tasks_per_service = [
            service.task_number for service in self.services
        ]
        self.D = sum(self.number_of_tasks_per_service)
        self.E = self.G.number_of_nodes()
        self.A = self.G.number_of_edges()
        self.arc_dic = {elem: i for i, elem in enumerate(self.G.edges)}
        self.arc_list = list(self.G.edges)
        self.origins = [
            origin for service in self.services for origin in service.origin_nodes
        ]
        self.minimum_flows = [
            min_flow for service in self.services for min_flow in service.min_flows
        ]
        self.maximum_flows = [
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
        self.shortest_paths = [
            [
                elem
                for target in self.G.nodes
                for elem in islice(
                    nx.algorithms.simple_paths.shortest_simple_paths(
                        self.G, self.origins[d], target
                    ),
                    self.limit,
                )
            ]
            for d in range(self.D)
        ]
        self.all_shortest_paths = [
            elem for path_list in self.shortest_paths for elem in path_list
        ]
        self.C = sum(len(path) for path in self.shortest_paths)
        self.D_path_number = [
            i for i, path_list in enumerate(self.shortest_paths) for _ in path_list
        ]
        self.D_path = [
            i for i, path_list in enumerate(self.shortest_paths) for _ in path_list
        ]
        self.m = max(self.maximum_flows)
        self.n = max(self.latencies)

        self.shortest_paths = [
            [
                list(
                    islice(
                        nx.algorithms.simple_paths.shortest_simple_paths(
                            self.G, self.origins[d], target
                        ),
                        self.limit,
                    )
                )
                for target in self.G.nodes
            ]
            for d in range(self.D)
        ]
        self.paths = [path[0] for l_path in self.shortest_paths for path in l_path]
        self.P = len(self.paths)
        self.p_to_d = [
            i
            for i in range(len(self.shortest_paths))
            for j in range(len(self.shortest_paths[i]))
        ]

        self.pd = [0] + [
            sum(len(elem) for elem in self.shortest_paths[:i])
            for i in range(1, len(self.shortest_paths) + 1)
        ]

        self.maximum_required_flow = max(self.maximum_flows)
        self.maximum_possible_latency = max(self.alpha) * sum(self.maximum_flows) + max(
            self.beta
        )

    def path_to_demand(self, p):
        return self.p_to_d[p]

    def delta(self, c, a):
        return int(
            self.arc_list[a]
            in [
                (self.paths[c][i], self.paths[c][i + 1])
                for i in range(len(self.paths[c]) - 1)
            ]
        )

    def _model_creation(self):
        self.model = pyo.ConcreteModel()

    def _set_creation(self):
        self.model.A = pyo.Set(initialize=range(self.A))
        self.model.P = pyo.Set(initialize=range(self.P))
        self.model.E = pyo.Set(initialize=range(self.E))
        self.model.S = pyo.Set(initialize=range(self.S))
        self.model.D = pyo.Set(initialize=range(self.D))

    def _variable_creation(self):
        self.model.L = pyo.Var(self.model.P, self.model.A, within=pyo.NonNegativeReals)
        self.model.Q = pyo.Var(self.model.P, within=pyo.NonNegativeReals)
        self.model.q = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.y = pyo.Var(
            self.model.S, self.model.E, within=pyo.NonNegativeReals, bounds=(0, 1)
        )
        self.model.l = pyo.Var(self.model.P, within=pyo.NonNegativeReals, bounds=(0, 1))
        self.model.z = pyo.Var(self.model.P, within=pyo.Binary)
        self.model.nul = pyo.Var(range(1))

    def _set_constraints(self):
        self.model.val_q = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: pyo.inequality(
                self.minimum_flows[d], model.q[d], self.maximum_flows[d]
            ),
        )
        self.model.res = pyo.Constraint(
            self.model.E,
            rule=lambda model, e: sum(
                model.y[s, e] * self.resources[s] for s in model.S
            )
            <= self.G.nodes[e]["available_resources"],
        )

        self.model.capa = pyo.Constraint(
            self.model.A,
            rule=lambda model, a: model.nul[0]
            + sum(self.delta(p, a) * model.Q[p] for p in model.P)
            <= self.G.edges[self.arc_list[a]]["capacity"],
        )

        self.model.lat = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            <= self.latencies[self.path_to_demand(p)],
        )

        self.model.ind1 = pyo.Constraint(
            self.model.P, rule=lambda model, p: model.l[p] <= model.z[p]
        )

        self.model.ind2 = pyo.Constraint(
            self.model.P, rule=lambda model, p: model.z[p] <= 1 + model.l[p] - self.eps
        )

        self.model.res_reach = pyo.Constraint(
            self.model.S, rule=lambda model, s: sum(model.y[s, e] for e in model.E) >= 1
        )

        self.model.reach_serv = pyo.Constraint(
            self.model.D, self.model.E, rule=self.reach_serv
        )

        self.model.qual1 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p] <= model.q[self.path_to_demand(p)],
        )

        self.model.qual2 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p]
            >= model.q[self.path_to_demand(p)]
            - self.maximum_required_flow * (1 - model.l[p]),
        )

        self.model.qual3 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p] <= self.maximum_required_flow * model.l[p],
        )

        self.model.lat1 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            <= self.alpha[a] * sum(self.delta(pp, a) * model.Q[pp] for pp in model.P)
            + self.beta[a],
        )

        self.model.lat2 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            >= self.alpha[a] * sum(self.delta(pp, a) * model.Q[pp] for pp in model.P)
            + self.beta[a]
            - self.maximum_possible_latency * (1 - model.z[p]),
        )

        self.model.lat3 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            <= self.maximum_possible_latency * model.z[p],
        )

    def gamma(self, p, e):
        return int(self.paths[p][-1] == e)

    def _set_objective(self):
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.q.values())
        )

    def _set_params(self):
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

    def reach_serv(self, model, d, e):
        return model.y[model.serv_number[d], e] <= sum(
            self.gamma(p, e) * model.l[p] for p in range(self.pd[d], self.pd[d + 1])
        )

    def run(
        self,
        options: dict = None,
        solving_time_limit: float = 600,
        thread_limit: int = 1,
        solver: str = "gurobi",
    ):
        self._model_parameters()
        self._model_creation()
        self._set_creation()
        self._variable_creation()
        self._set_params()
        self._set_constraints()
        self._set_objective()

        opt = pyo.SolverFactory(solver)
        if solver == "gurobi":
            opt.options["TimeLimit"] = solving_time_limit
            opt.options["threads"] = thread_limit
        else:
            assert False, "Not implemented yet."

        try:
            solver_result = opt.solve(self.model)
            (
                self.runtime,
                self.lower_bound,
                self.upper_bound,
                self.solving_status,
            ) = get_result_informations(solver_result)

        except ValueError:
            self.runtime, self.lower_bound, self.upper_bound = solving_time_limit, 0, 0
