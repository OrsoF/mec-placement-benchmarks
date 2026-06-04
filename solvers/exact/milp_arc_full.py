from core.generic_solver import GenericSolver
from core.problem_classes.service import Service
from core.problem_classes.problem_instance import ProblemInstance
from core.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from data_io.experiment_io import get_result_informations
from solvers.arc_formulation import add_arc_constraints
import pyomo.environ as pyo
import time
import networkx as nx
import numpy as np


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
        ) # x

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.UnitInterval
        )

        self.model.p = pyo.Var(
            self.model.total_number_of_demands, within=pyo.NonNegativeReals
        )

        self.model.P = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeReals,
        )

        self.model.L = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeReals,
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
        add_arc_constraints(
            self.model,
            self.model.total_number_of_demands,
            self.model.arc_number,
            self.model.node_number,
            self.model.number_of_services,
            self.model.demand_arc_transit,
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(
            sense=pyo.maximize, rule=lambda model: sum(model.p.values())
        )

    def _set_initial_value(self, initial_value: tuple):
        x_variable, y_variable, q_variable = initial_value
        for demand_index in range(self.total_number_of_demands):
            self.model.p[demand_index].Start = q_variable[demand_index]
            for arc_index in range(self.arc_number):
                self.model.demand_arc_transit[demand_index, arc_index].Start = (
                    x_variable[demand_index, arc_index]
                )
        for service_index in range(self.number_of_services):
            for node_index in range(self.node_number):
                self.model.y[service_index, node_index].Start = y_variable[
                    service_index, node_index
                ]

    def run(
        self,
        thread_limit: int = 1,
        initial_value=None,
    ):
        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if initial_value is not None:
            self._set_initial_value(initial_value)

        opt = pyo.SolverFactory("gurobi")
        opt.options["threads"] = 1
        opt.options['Heuristics'] = 0
        solver_result = opt.solve(self.model)

        (
            self.runtime,
            self.lower_bound,
            self.upper_bound,
            self.solving_status,
        ) = get_result_informations(solver_result)

        self._set_arc_model_solution(
            self.model.demand_arc_transit,
            self.model.y,
            self.model.p,
            self.total_number_of_demands,
            self.arc_number,
            self.number_of_services,
            self.node_number,
        )
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
