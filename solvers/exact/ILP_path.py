from core.generic_solver import GenericSolver
from core.problem_classes.service import Service
from core.problem_classes.problem_instance import ProblemInstance
from data_io.experiment_io import get_result_informations
from solvers.path_formulation import (
    add_path_constraints,
    add_path_params,
    enumerate_candidate_paths,
)

import pyomo.environ as pyo
import networkx as nx
import numpy as np
import time


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, maximum_path_number: int = 3):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.maximum_path_number = maximum_path_number
        self.solver_name = "ILP_path"
        # print(f"Number of SP: {maximum_path_number}")

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

        (
            self.paths,
            self.all_shortest_paths,
            self.D_path_number,
            self.path_start,
        ) = enumerate_candidate_paths(self.G, self.origins, self.maximum_path_number)
        self.C = len(self.all_shortest_paths)
        self.m = max(self.Pdmax)
        self.n = max(self.latencies)

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

        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={
                a: self.G.edges[edge]["capacity"] for a, edge in enumerate(self.G.edges)
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

        add_path_params(
            self.model,
            self.paths,
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
        opt.options["Heuristics"] = 0
        opt.options["MemLimit"] = 8

        try:
            solver_result = opt.solve(self.model)
            (
                solver_runtime,
                lower_bound,
                _,
                self.solving_status,
            ) = get_result_informations(solver_result)
            self.score = float(lower_bound)
        except:
            self._set_problem_baseline_solution()
            self.runtime = time.time() - start_time
            self.build_time = 0.0
            return self

        self._set_path_model_solution(
            self.model.l,
            self.model.y,
            self.model.p,
            self.all_shortest_paths,
            self.D_path_number,
            self.arc_dic,
            self.demand_nb,
            self.edge_nb,
            self.service_nb,
            self.node_nb,
        )
        self.score = float(np.sum(self.q_var))
        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
        return self
