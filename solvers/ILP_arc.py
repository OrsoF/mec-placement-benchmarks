from utils.generic_solver import GenericSolver
from problem_classes.service import Service
from problem_classes.probleminstance import ProblemInstance
from utils.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from utils.data_management import get_result_informations
import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import os
import numpy as np

from solvers.heuristic import Solver as Heursolv
from solvers.rl_PPO import Solver as RlPPO


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "ILP_arc"
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

        # Solution matrices
        self.demand_arc_transit: np.ndarray  # x matrix
        self.service_node_matrix: np.ndarray  # y matrix
        self.quantity_matrix: np.ndarray  # q matrix

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.total_number_of_demands = pyo.Set(
            initialize=range(self.total_number_of_demands)
        )
        self.model.number_of_services = pyo.Set(
            initialize=range(self.number_of_services)
        )
        self.model.node_number = pyo.Set(initialize=range(self.node_number))
        self.model.arc_number = pyo.Set(initialize=range(self.arc_number))

    def _build_variables(self):
        self.model.demand_arc_transit = pyo.Var(
            self.model.total_number_of_demands, self.model.arc_number, within=pyo.Binary
        )

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.Binary
        )

        self.model.p = pyo.Var(
            self.model.total_number_of_demands, within=pyo.NonNegativeIntegers
        )

        self.model.P = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

        self.model.L = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.total_number_of_demands,
            initialize={
                d: self.origins[d] for d in range(self.total_number_of_demands)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.total_number_of_demands,
            initialize={
                d: self.min_flows[d] for d in range(self.total_number_of_demands)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.total_number_of_demands,
            initialize={
                d: self.max_flows[d] for d in range(self.total_number_of_demands)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.total_number_of_demands,
            initialize={
                d: self.latencies[d] for d in range(self.total_number_of_demands)
            },
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.number_of_services,
            initialize={
                s: self.available_resources[s] for s in range(self.number_of_services)
            },
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

        self.model.alpha = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.alphas[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.arc_number,
            initialize={edge: self.betas[edge] for edge in self.model.arc_number},  # type: ignore
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.serv_number = pyo.Param(
            self.model.total_number_of_demands,
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
                u: get_encoded_outgoing_arc(self.G, self.arc_index_dictionary, u)
                for u in range(self.node_number)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.inc_arcs = pyo.Param(
            self.model.node_number,
            initialize={
                u: get_encoded_incoming_arc(self.G, self.arc_index_dictionary, u)
                for u in range(self.node_number)
            },
            mutable=False,
            within=pyo.Any,
        )
        self.model.ind_u_origin = pyo.Param(
            self.model.node_number,
            self.model.total_number_of_demands,
            initialize={
                (u, d): int(u == self.model.origins[d])
                for u in range(self.node_number)
                for d in range(self.total_number_of_demands)
            },
            mutable=False,
            within=pyo.Binary,
        )

        self.model.node_resources = pyo.Param(
            self.model.node_number,
            initialize=self.available_resources,
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
            self.model.total_number_of_demands,
            rule=lambda model, d: pyo.inequality(
                model.Pdmin[d], model.p[d], model.Pdmax[d]
            ),
        )

        self.model.valeurs_P = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a] <= model.Pdmax[d],
        )

        self.model.conservation_constraint = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.node_number,
            rule=lambda model, d, u: sum(
                model.demand_arc_transit[d, arc] for arc in model.out_arcs[u]
            )
            - sum(model.demand_arc_transit[d, arc] for arc in model.inc_arcs[u])
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
            rule=lambda model, a: sum(
                [model.P[d, a] for d in self.model.total_number_of_demands]
            )
            <= model.arc_capacity[a],
        )

        self.model.latency_arc_1 = pyo.Constraint(
            self.model.total_number_of_demands,
            rule=lambda model, d: sum(model.L[d, ap] for ap in model.arc_number)
            <= model.latencies[d],
        )
        self.model.latency_arc_2 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a]
            <= model.alpha[a]
            * sum(model.P[dp, a] for dp in self.model.total_number_of_demands)
            + model.beta[a],
        )

        self.model.latency_arc_3 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a]
            >= model.alpha[a]
            * sum(model.P[dp, a] for dp in self.model.total_number_of_demands)
            + model.beta[a]
            - model.N * (1 - model.demand_arc_transit[d, a]),
        )
        self.model.latency_arc_4 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.L[d, a]
            <= model.N * model.demand_arc_transit[d, a],
        )

        self.model.power_1 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a] <= model.p[d],
        )

        self.model.power_2 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a]
            >= model.p[d] - model.M * (1 - model.demand_arc_transit[d, a]),
        )

        self.model.power_3 = pyo.Constraint(
            self.model.total_number_of_demands,
            self.model.arc_number,
            rule=lambda model, d, a: model.P[d, a]
            <= model.M * model.demand_arc_transit[d, a],
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.p.values())
        )

    def _set_initial_value(self, initial_value: tuple):
        x_var, y_var, q_var = initial_value
        for demand_index in range(self.total_number_of_demands):
            self.model.p[demand_index].value = q_var[demand_index]
            for arc_index in range(self.arc_number):
                self.model.demand_arc_transit[demand_index, arc_index].value = x_var[
                    demand_index, arc_index
                ]
        for service_index in range(self.number_of_services):
            for node_index in range(self.node_number):
                self.model.y[service_index, node_index].value = y_var[
                    service_index, node_index
                ]

    def set_heuristic_placement(self):
        cloud_server = np.argmax(
            self.problem_instance.network_graph.available_resources
        )
        for service_index in range(self.problem_instance.number_of_service):
            for server_index in range(self.nx_graph.number_of_nodes()):
                if server_index == cloud_server:
                    self.model.y[service_index, server_index].fix(1)
                # else:
                #     self.model.y[service_index, server_index].fix(0)

    def set_heuristic_solution(self):
        heuristic_solver = Heursolv(self.problem_instance)
        heuristic_solver.run()
        self._set_initial_value(
            (heuristic_solver.x_var, heuristic_solver.y_var, heuristic_solver.q_var)
        )

    def set_rl_solution(self):

        solver = RlPPO(problem_instance=self.problem_instance)
        solver.run()

        best_state = solver.env.best_solution[0]

        x_var, y_var, q_var = solver.env._state_to_variables(best_state)
        self._set_initial_value((x_var, y_var, q_var))

    def run(
        self,
        initial_value: tuple = None,
        options: dict = {},
        heuristic_placement: bool = False,
        heuristic_solution: bool = False,
        rl_solution: bool = False,
    ):
        start_time = time.time()

        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if initial_value is not None:
            self._set_initial_value(initial_value)

        if heuristic_placement:
            self.set_heuristic_placement()

        if heuristic_solution:
            self.set_heuristic_solution()

        if rl_solution:
            self.set_rl_solution()

        opt = pyo.SolverFactory("gurobi")
        # opt.options["threads"] = 1
        opt.options["Heuristics"] = 0
        opt.options["MemLimit"] = 8

        for key, value in options.items():
            opt.options[key] = value

        solver_result = opt.solve(self.model)
        solver_runtime = get_result_informations(solver_result)[0]

        self.x_var = np.array(
            [
                [
                    self.model.demand_arc_transit[i, j].value
                    for j in range(self.arc_number)
                ]
                for i in range(self.total_number_of_demands)
            ]
        )
        self.y_var = np.array(
            [
                [self.model.y[i, j].value for j in range(self.node_number)]
                for i in range(self.number_of_services)
            ]
        )
        self.q_var = np.array(
            [self.model.p[dd].value for dd in range(self.total_number_of_demands)]
        )

        try:
            self.score = np.sum(self.q_var)
        except:
            self.score = 0
            
        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
