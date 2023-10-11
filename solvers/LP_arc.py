from utils.instance_solver import InstanceSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.pyomo_utils import get_info_from_results
from utils.graph_neighbors import get_encoded_incoming_arc, get_encoded_outgoing_arc

import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os


class Solver(InstanceSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
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
            initialize={u: self.G.nodes[u]["resources"] for u in range(self.U)},
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
        self.model.valeurs_p = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: pyo.inequality(
                model.Pdmin[d], model.p[d], model.Pdmax[d]
            ),
        )
        self.model.valeurs_P = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.P[d, a] <= model.Pdmax[d],
        )
        self.model.conservation_constraint = pyo.Constraint(
            self.model.D,
            self.model.U,
            rule=lambda model, d, u: sum(model.x[d, arc] for arc in model.out_arcs[u])
            - sum(model.x[d, arc] for arc in model.inc_arcs[u])
            == model.ind_u_origin[u, d] - model.y[model.serv_number[d], u],
        )
        self.model.resources_constraint = pyo.Constraint(
            self.model.U,
            rule=lambda model, u: sum(
                [model.y[s, u] * model.resources[s] for s in model.S]
            )
            <= model.node_resources[u],
        )
        self.model.capacity_constraint = pyo.Constraint(
            self.model.A,
            rule=lambda model, a: sum([model.P[d, a] for d in model.D])
            <= model.arc_capacity[a],
        )
        self.model.latency_arc_1 = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: sum(model.L[d, ap] for ap in model.A)
            <= model.latencies[d],
        )
        self.model.latency_arc_2 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.L[d, a]
            <= model.alpha[a] * sum(model.P[dp, a] for dp in model.D) + model.beta[a],
        )
        self.model.latency_arc_3 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.L[d, a]
            >= model.alpha[a] * sum(model.P[dp, a] for dp in model.D)
            + model.beta[a]
            - model.N * (1 - model.x[d, a]),
        )
        self.model.latency_arc_4 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.L[d, a] <= model.N * model.x[d, a],
        )

        self.model.power_1 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.P[d, a] <= model.p[d],
        )
        self.model.power_2 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.P[d, a]
            >= model.p[d] - model.M * (1 - model.x[d, a]),
        )
        self.model.power_3 = pyo.Constraint(
            self.model.D,
            self.model.A,
            rule=lambda model, d, a: model.P[d, a] <= model.M * model.x[d, a],
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.p.values())
        )

    def _solve_problem(self):
        opt = pyo.SolverFactory(self.solver)
        for key in self.options.keys():
            opt.options[key] = self.options[key]
        try:
            solver_result = opt.solve(self.model)

            results_informations = [
                "Time: ",
                "Lower bound: ",
                "Upper bound: ",
                "Termination condition: ",
            ]
            self.results = {}
            for elem in results_informations:
                self.results[elem] = get_info_from_results(solver_result, elem)
        except ValueError:
            self.results = {
                "Time: ": self.options["TimeLimit"],
                "Lower bound: ": 0,
                "Upper bound: ": 0,
                "Termination condition: ": False,
            }

    def run(self, solver: str = "gurobi", options: dict = {"time": 3600, "threads": 1}):
        self.solver = solver
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
        self.lower_bound = self.results["Lower bound: "]
        self.upper_bound = self.results["Upper bound: "]
