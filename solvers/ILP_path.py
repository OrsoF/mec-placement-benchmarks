from utils.generic_solver import GenericSolver
from problem_classes.servicepackage import ServicePackage
from problem_classes.networkgraph import NetworkGraph
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.data_management import get_info_from_results
from utils.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from itertools import islice
from utils.data_management import get_result_informations
from networkx.readwrite.gml import read_gml

import pyomo.environ as pyo
import networkx as nx
import numpy as np
import os
import time


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, maximum_path_number: int = 3):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.maximum_path_number = maximum_path_number
        self.solver_name = "ILP_path"
        print(f"Number of SP: {maximum_path_number}")

        # Graphs
        self.network_graph = self.problem_instance.network_graph
        self.nx_graph = self.network_graph.nx_graph

    def _model_parameters(self):
        # Service parameters
        self.number_of_services = self.problem_instance.number_of_service
        self.demand_per_service = self.problem_instance.demand_per_service
        self.total_number_of_demands = self.problem_instance.total_number_of_demands
        self.needed_resources = self.problem_instance.needed_resources

        self.origins = self.problem_instance.origins
        self.min_flows = self.problem_instance.min_flows
        self.max_flows = self.problem_instance.max_flows
        self.latencies = self.problem_instance.latencies

        # Graph parameters
        self.arc_index_dictionary = self.network_graph.arc_index_dictionary
        self.arc_list = self.network_graph.arc_list
        self.alphas = self.network_graph.alphas
        self.betas = self.network_graph.betas
        self.node_number = self.nx_graph.number_of_nodes()
        self.arc_number = self.nx_graph.number_of_edges()
        self.capacities = self.network_graph.capacities
        self.available_resources = self.network_graph.available_resources

        self.service_nb = len(self.services)
        self.task_nb_list = [service.task_number for service in self.services]
        self.demand_nb = sum(self.task_nb_list)
        self.node_nb = self.G.number_of_nodes()
        self.edge_nb = self.G.number_of_edges()
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

        self.paths = [
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
            for d in range(self.demand_nb)
        ]

        self.all_shortest_paths = [
            elem for path_list in self.paths for elem in path_list
        ]
        self.C = sum(len(elem) for elem in self.paths)
        self.m = max(self.Pdmax)
        self.n = max(self.latencies)
        self.D_path_number = [
            i for i, path_list in enumerate(self.paths) for _ in path_list
        ]

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.demand_nb))
        self.model.S = pyo.Set(initialize=range(self.service_nb))
        self.model.U = pyo.Set(initialize=range(self.node_nb))
        self.model.A = pyo.Set(initialize=range(self.edge_nb))
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
            initialize={d: self.origins[d] for d in range(self.demand_nb)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.D,
            initialize={d: self.Pdmin[d] for d in range(self.demand_nb)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.D,
            initialize={d: self.Pdmax[d] for d in range(self.demand_nb)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.D,
            initialize={d: self.latencies[d] for d in range(self.demand_nb)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.S,
            initialize={s: self.resources[s] for s in range(self.service_nb)},
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
            initialize={
                u: self.G.nodes[u]["available_resources"] for u in range(self.node_nb)
            },
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
                for a in range(self.edge_nb)
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
            range(self.demand_nb + 1),
            initialize={
                d: sum(len(elem) for elem in self.paths[:d])
                for d in range(self.demand_nb + 1)
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
                for u in range(self.node_nb)
            },
        )
        D_path = [i for i, path_list in enumerate(self.paths) for _ in path_list]
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

    def run(
        self,
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
        # opt.options["threads"] = 1
        opt.options["Heuristics"] = 0
        opt.options["MemLimit"] = 8
        opt.options["TimeLimit"] = 600

        try:
            print("Start solving process...")
            solver_result = opt.solve(self.model)
            (
                solver_runtime,
                lower_bound,
                _,
                self.solving_status,
            ) = get_result_informations(solver_result)

        except ValueError:
            solver_runtime, lower_bound, _ = 600, 0, 0

        # self.lambda_variable = np.array(
        #     [
        #         [
        #             self.model.demand_arc_transit[i, j].value
        #             for j in range(self.arc_number)
        #         ]
        #         for i in range(self.total_number_of_demands)
        #     ]
        # )
        # self.y_variable = np.array(
        #     [
        #         [self.model.y[i, j].value for j in range(self.node_number)]
        #         for i in range(self.number_of_services)
        #     ]
        # )
        # self.q_variable = np.array(
        #     [self.model.p[dd].value for dd in range(self.total_number_of_demands)]
        # )

        self.runtime = time.time() - start_time
        self.score = float(lower_bound)
        self.build_time = self.runtime - float(solver_runtime)
