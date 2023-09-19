from utils.instance_solver import InstanceSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.pyomo_utils import get_info_from_results
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc
from itertools import islice

import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os


class Solver(InstanceSolver):
    def __init__(self, problem_instance: ProblemInstance, maximum_path_number: int = 1):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.maximum_path_number = maximum_path_number
        self.solver_name = "ILP_path"

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
            resources
            for service in self.services
            for resources in service.needed_resources
        ]
        self.alpha = [self.G.edges[edge]["alpha"] for edge in self.G.edges]
        self.beta = [self.G.edges[edge]["beta"] for edge in self.G.edges]

        self.shortest_paths = [
            [
                path
                for target in self.G.nodes
                for path in islice(
                    nx.algorithms.simple_paths.shortest_simple_paths(
                        self.G, self.origins[d], target
                    ),
                    self.maximum_path_number,
                )
            ]
            for d in range(self.D)
        ]

        self.all_shortest_paths = [
            elem for path_list in self.shortest_paths for elem in path_list
        ]
        self.C, C_demand = sum(len(elem) for elem in self.shortest_paths), [
            len(elem) for elem in self.shortest_paths
        ]
        self.m = max(self.Pdmax)
        self.n = max(self.latencies)
        self.D_path_number = [
            i for i, path_list in enumerate(self.shortest_paths) for path in path_list
        ]

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.D))
        self.model.S = pyo.Set(initialize=range(self.S))
        self.model.U = pyo.Set(initialize=range(self.U))
        self.model.A = pyo.Set(initialize=range(self.A))
        self.model.C = pyo.Set(initialize=range(self.C))

    def _build_variables(self):
        self.model.l = pyo.Var(self.model.C, within=pyo.Binary)
        self.model.y = pyo.Var(self.model.S, self.model.U, within=pyo.Binary)
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeIntegers)
        self.model.P = pyo.Var(self.model.C, within=pyo.NonNegativeIntegers)
        self.model.L = pyo.Var(
            self.model.C, self.model.A, within=pyo.NonNegativeIntegers
        )

        self.model.nul = pyo.Var(range(1), within=pyo.Binary)
        self.model.nul[0].fix(0)  # type: ignore

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
        self.model.Pdmax_path = pyo.Param(
            self.model.C,
            initialize={c: self.Pdmax[self.D_path_number[c]] for c in range(self.C)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latency_path = pyo.Param(
            self.model.C,
            initialize={
                c: self.latencies[self.D_path_number[c]] for c in range(self.C)
            },
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.node_resources = pyo.Param(
            self.model.U,
            initialize={u: self.G.nodes[u]["resources"] for u in range(self.U)},
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

        self.model.delta = pyo.Param(
            self.model.C,
            self.model.A,
            initialize={
                (c, a): (
                    self.arc_list[a]
                    in [
                        (
                            self.all_shortest_paths[c][i],
                            self.all_shortest_paths[c][i + 1],
                        )
                        for i in range(len(self.all_shortest_paths[c]) - 1)
                    ]
                )
                for a in range(self.A)
                for c in range(self.C)
            },
            mutable=False,
            within=pyo.Binary,
        )
        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={
                a: self.G.edges[edge]["capacity"] for a, edge in enumerate(self.G.edges)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.all_shortest_paths_arc = pyo.Param(
            self.model.C,
            initialize={
                c: [
                    self.arc_dic[
                        (
                            self.all_shortest_paths[c][i],
                            self.all_shortest_paths[c][i + 1],
                        )
                    ]
                    for i in range(len(self.all_shortest_paths[c]) - 1)
                ]
                for c in range(self.C)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.path_demand = pyo.Param(
            range(self.D + 1),
            initialize={
                d: sum(len(elem) for elem in self.shortest_paths[:d])
                for d in range(self.D + 1)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        # Serv_number is a list : when iterating on all tasks with an index i, it gives the service associated to the i-th task.
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

        self.model.ind_destination = pyo.Param(
            self.model.C,
            self.model.U,
            initialize={
                (c, u): self.all_shortest_paths[c][-1] == u
                for c in range(self.C)
                for u in range(self.U)
            },
        )
        D_path = [
            i for i, path_list in enumerate(self.shortest_paths) for path in path_list
        ]
        self.model.D_path = pyo.Param(
            self.model.C,
            initialize={c: D_path[c] for c in range(self.C)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

    def _build_constraints(self):
        self.model.valeurs_p = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: pyo.inequality(
                model.Pdmin[d], model.p[d], model.Pdmax[d]
            ),
        )
        self.model.valeurs_P = pyo.Constraint(
            self.model.C, rule=lambda model, c: model.P[c] <= model.Pdmax_path[c]
        )
        self.model.valeurs_L = pyo.Constraint(
            self.model.C,
            self.model.A,
            rule=lambda model, c, a: model.L[c, a] <= model.latency_path[c],
        )
        self.model.resources_constraint = pyo.Constraint(
            self.model.U,
            rule=lambda model, u: sum(
                model.resources[s] * model.y[s, u] for s in model.S
            )
            <= model.node_resources[u],
        )
        self.model.capacity = pyo.Constraint(
            self.model.A,
            rule=lambda model, a: model.nul[0]
            + sum(float(model.delta[c, a]) * model.P[c] for c in model.C)
            <= model.arc_capacity[a],
        )
        self.model.latency = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: sum(
                model.L[c, a]
                for c in range(model.path_demand[d], model.path_demand[d + 1])
                for a in model.all_shortest_paths_arc[c]
            )
            <= model.latencies[d],
        )
        self.model.realization = pyo.Constraint(
            self.model.S, rule=lambda model, s: sum(model.y[s, u] for u in model.U) >= 1
        )
        self.model.reach_server = pyo.Constraint(
            self.model.D,
            self.model.U,
            rule=lambda model, d, u: model.y[model.serv_number[d], u]
            <= sum(
                float(model.ind_destination[c, u]) * model.l[c]
                for c in range(model.path_demand[d], model.path_demand[d + 1])
            ),
        )
        self.model.flow_rule_1 = pyo.Constraint(
            self.model.C, rule=lambda model, c: model.P[c] <= model.p[model.D_path[c]]
        )
        self.model.flow_rule_2 = pyo.Constraint(
            self.model.C, rule=lambda model, c: model.P[c] <= self.m * model.l[c]
        )
        self.model.flow_rule_3 = pyo.Constraint(
            self.model.C,
            rule=lambda model, c: model.P[c]
            >= model.p[model.D_path[c]] - self.m * (1 - model.l[c]),
        )

        self.model.latency_rule_1 = pyo.Constraint(
            self.model.C,
            self.model.A,
            rule=lambda model, c, a: model.L[c, a]
            <= model.alpha[a]
            * sum(float(model.delta[cp, a]) * model.P[cp] for cp in model.C)
            + model.beta[a],
        )
        self.model.latency_rule_2 = pyo.Constraint(
            self.model.C,
            self.model.A,
            rule=lambda model, c, a: model.L[c, a] <= self.n * model.l[c],
        )
        self.model.latency_rule_3 = pyo.Constraint(
            self.model.C,
            self.model.A,
            rule=lambda model, c, a: model.L[c, a]
            >= model.alpha[a]
            * sum(float(model.delta[cp, a]) * model.P[cp] for cp in model.C)
            + model.beta[a]
            - self.n * (1 - model.l[c]),
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.p.values())
        )

    def _solve_problem(self):
        opt = pyo.SolverFactory(self.solver)
        for key in self.options.keys():
            opt.options[key] = self.options[key]
        resultat = opt.solve(self.model)
        res = ["Time: ", "Lower bound: ", "Upper bound: ", "Termination condition: "]
        results = {}
        for elem in res:
            results[elem] = get_info_from_results(resultat, elem)
        self.results = results

    def run(self, solver: str = "gurobi", options: dict = {}):
        self.solver = solver
        if options is None:
            self.options = {"time": 3600, "threads": 1}
        else:
            self.options = options

        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()
        self._solve_problem()

        self.runtime = self.results["Time: "]
        self.objective_value = self.results["Lower bound: "]
