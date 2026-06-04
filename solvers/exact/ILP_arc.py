from core.generic_solver import GenericSolver
from core.pyomo_model import configure_gurobi, objective_value, problem_arrays
from core.problem_classes.service import Service
from core.problem_classes.problem_instance import ProblemInstance
from core.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from data_io.experiment_io import get_result_informations
from solvers.arc_formulation import add_arc_constraints
import pyomo.environ as pyo
import time
import networkx as nx
import numpy as np

from solvers.heuristic.heuristic import Solver as Heursolv
from solvers.rl.ppo import Solver as RlPPO


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.services: list[Service] = self.problem_instance.service_package.services
        self.G: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "ILP_arc"
        self.network_graph = self.problem_instance.network_graph
        self.nx_graph = self.network_graph.nx_graph

    def _model_parameters(self):
        a = problem_arrays(self.problem_instance)
        self.number_of_services = self.S = a.S
        self.total_number_of_demands = self.D = a.D
        self.node_number = self.U = a.U
        self.arc_number = self.A = a.A
        self.demand_per_service = self.problem_instance.demand_per_service
        self.needed_resources = self.resources = a.resources
        self.origins = a.origins
        self.min_flows = self.qmin = a.qmin
        self.max_flows = self.qmax = a.qmax
        self.latencies = a.latencies
        self.arc_index_dictionary = self.arc_dic = a.arc_index
        self.arc_list = a.arc_list
        self.alphas = self.alpha = a.alpha
        self.betas = self.beta = a.beta
        self.capacities = a.capacities
        self.available_resources = a.node_resources

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
                s: self.needed_resources[s] for s in range(self.number_of_services)
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

        print(cloud_server)

        for service in range(self.problem_instance.number_of_service):
            self.model.y[service, cloud_server].fix(1)

    def set_heuristic_solution(self):
        heuristic_solver = Heursolv(self.problem_instance)
        heuristic_solver.run()
        self._set_initial_value(
            (heuristic_solver.x_var, heuristic_solver.y_var, heuristic_solver.q_var)
        )

    def set_rl_solution(self):

        solver = RlPPO(self.problem_instance)
        solver.run()

        best_state = solver.env.best_state_record[0]

        x_var, y_var, q_var = solver.env._state_to_variables(best_state)
        self._set_initial_value((x_var, y_var, q_var))

    def run(
        self,
        initial_value: tuple = None,
        options: dict | None = None,
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

        opt = configure_gurobi(
            pyo.SolverFactory("gurobi"),
            {"Heuristics": 0, "MemLimit": 8, **(options or {})},
        )

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
        self._set_output_variables(self.x_var, self.y_var, self.q_var)

        try:
            self.score = objective_value(self.q_var)
        except:
            self.score = 0
            
        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
