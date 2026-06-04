from core.generic_solver import GenericSolver
from core.pyomo_model import objective_value, problem_arrays
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
        a = problem_arrays(self.problem_instance)
        self.number_of_services = self.service_nb = a.S
        self.total_number_of_demands = self.demand_nb = a.D
        self.node_number = self.node_nb = a.U
        self.arc_number = self.edge_nb = a.A
        self.demand_per_service = self.task_nb_list = self.problem_instance.demand_per_service
        self.needed_resources = self.resources = a.resources
        self.origins = a.origins
        self.min_flows = self.Pdmin = a.qmin
        self.max_flows = self.Pdmax = a.qmax
        self.latencies = a.latencies
        self.arc_index_dictionary = self.arc_dic = a.arc_index
        self.arc_list = a.arc_list
        self.alphas = self.alpha = a.alpha
        self.betas = self.beta = a.beta
        self.capacities = a.capacities
        self.available_resources = a.node_resources

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
        self.score = objective_value(self.q_var)
        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
        return self
