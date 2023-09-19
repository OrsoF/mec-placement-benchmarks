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
        self.solver_name = "ILP_arc"

    def _model_parameters(self):
        self.number_of_services = len(self.services)
        self.demand_per_service = [service.task_number for service in self.services]
        self.total_demands = sum(self.demand_per_service)
        self.node_number = self.G.number_of_nodes()
        self.arc_number = self.G.number_of_edges()
        self.arc_dic = {arc: i for i, arc in enumerate(self.G.edges)}
        self.arc_list = list(self.G.edges)
        self.origins = [
            origin for service in self.services for origin in service.origin_nodes
        ]
        self.min_flows = [
            min_flow for service in self.services for min_flow in service.min_flows
        ]
        self.max_flows = [
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
        self.model.total_demands = pyo.Set(initialize=range(self.total_demands))
        self.model.number_of_services = pyo.Set(
            initialize=range(self.number_of_services)
        )
        self.model.node_number = pyo.Set(initialize=range(self.node_number))
        self.model.arc_number = pyo.Set(initialize=range(self.arc_number))

    def _build_variables(self):
        self.model.x = pyo.Var(
            self.model.total_demands, self.model.arc_number, within=pyo.Binary
        )

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.Binary
        )

        self.model.p = pyo.Var(self.model.total_demands, within=pyo.NonNegativeIntegers)

        self.model.P = pyo.Var(
            self.model.total_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

        self.model.L = pyo.Var(
            self.model.total_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.total_demands,
            initialize={d: self.origins[d] for d in range(self.total_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.total_demands,
            initialize={d: self.min_flows[d] for d in range(self.total_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.total_demands,
            initialize={d: self.max_flows[d] for d in range(self.total_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.total_demands,
            initialize={d: self.latencies[d] for d in range(self.total_demands)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.number_of_services,
            initialize={s: self.resources[s] for s in range(self.number_of_services)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.alpha = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.alpha[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.beta[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.serv_number = pyo.Param(
            self.model.total_demands,
            initialize={
                sum(service.task_number for service in self.services[:service_index])
                + j: service_index
                for service_index, service in enumerate(self.services)
                for j in range(service.task_number)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.out_arcs = pyo.Param(
            self.model.node_number,
            initialize={
                u: get_encoded_outgoing_arc(self.G, self.arc_dic, u)
                for u in range(self.node_number)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.node_number,
            initialize={
                u: get_encoded_incoming_arc(self.G, self.arc_dic, u)
                for u in range(self.node_number)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.ind_u_origin = pyo.Param(
            self.model.node_number,
            self.model.total_demands,
            initialize={
                (u, d): int(u == self.model.origins[d])
                for u in range(self.node_number)
                for d in range(self.total_demands)
            },
            mutable=False,
            within=pyo.Binary,
        )

        self.model.node_resources = pyo.Param(
            self.model.node_number,
            initialize={
                u: self.G.nodes[u]["resources"] for u in range(self.node_number)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.arc_capacity = pyo.Param(
            self.model.arc_number,
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
            initialize=max(self.max_flows),
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

    def _build_constraints(self):
        self.model.valeurs_p = pyo.Constraint(
            self.model.total_demands,
            rule=lambda model, d: pyo.inequality(
                model.Pdmin[d], model.p[d], model.Pdmax[d]
            ),
        )

        self.model.valeurs_P = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a] <= model.Pdmax[d],
        )

        self.model.conservation_constraint = pyo.Constraint(
            self.model.total_demands,
            self.model.node_number,
            rule=lambda model, d, u: sum(model.x[d, arc] for arc in model.out_arcs[u])
            - sum(model.x[d, arc] for arc in model.inc_arcs[u])
            == model.ind_u_origin[u, d] - model.y[model.serv_number[d], u],
        )

        self.model.resources_constraint = pyo.Constraint(
            self.model.node_number,
            rule=lambda model, u: sum(
                [model.y[s, u] * model.resources[s] for s in model.number_of_services]
            )
            <= model.node_resources[u],
        )

        self.model.capacity_constraint = pyo.Constraint(
            self.model.arc_number,
            rule=lambda model, a: sum([model.P[d, a] for d in model.total_demands])
            <= model.arc_capacity[a],
        )

        self.model.latency_arc_1 = pyo.Constraint(
            self.model.total_demands,
            rule=lambda model, d: sum(model.L[d, ap] for ap in model.arc_number)
            <= model.latencies[d],
        )
        self.model.latency_arc_2 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a]
            <= model.alpha[a] * sum(model.P[dp, a] for dp in model.total_demands)
            + model.beta[a],
        )

        self.model.latency_arc_3 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a]
            >= model.alpha[a] * sum(model.P[dp, a] for dp in model.total_demands)
            + model.beta[a]
            - model.N * (1 - model.x[d, a]),
        )
        self.model.latency_arc_4 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a] <= model.N * model.x[d, a],
        )

        self.model.power_1 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a] <= model.p[d],
        )

        self.model.power_2 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a]
            >= model.p[d] - model.M * (1 - model.x[d, a]),
        )

        self.model.power_3 = pyo.Constraint(
            self.model.total_demands,
            self.model.arc_number,
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
        resultat = opt.solve(self.model)
        res = ["Time: ", "Lower bound: ", "Upper bound: ", "Termination condition: "]
        results = {}
        for elem in res:
            results[elem] = get_info_from_results(resultat, elem)
        self.results = results

    def run(self, options: dict, solver: str = "gurobi"):
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
        self.objective_value = self.results["Lower bound: "]
