from problem_classes.probleminstance import ProblemInstance
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from solvers.base_heuristic import Solver as base_heuristic

np.random.seed(0)
import math


class Solver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.runtime: float
        self.lower_bound: float
        self.upper_bound: float
        self.solver_name = "column_generation"

        # Graphs
        self.network_graph = self.problem_instance.network_graph
        self.nx_graph = self.network_graph.nx_graph

        self._model_parameters()

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

    def generate_initial_solution(self):
        base_heuristic_solver = base_heuristic(self.problem_instance)
        base_heuristic_solver.run()
        y = base_heuristic_solver.service_node_matrix
        x = base_heuristic_solver.x
        p = base_heuristic_solver.quantity_matrix
        return y, x, p

    def master_problem(self, instance, pattern):
        pass

    def subproblem(self, instance, pattern):
        pass

    def col_gen(self):
        # Generate initial solution
        x, y, p = self.generate_initial_solution()
        objVal_history = []
        while True:
            master_problem = self.master_problem(ins_, patterns)
            master_problem.optimize()
            objVal_history.append(master_problem.objVal)
            dual_variables = np.array(
                [constraint.pi for constraint in master_problem.getConstrs()]
            )
            subproblem = define_subproblem(ins_, dual_variables)
            subproblem.optimize()
            if subproblem.objVal < 1 + 1e-2:
                break
            patterns.append([i.x for i in subproblem.getVars()])
        print_solution(master_problem, patterns)
        print(
            "Total number of rolls used: ",
            int(np.array([math.ceil(i.x) for i in master_problem.getVars()]).sum()),
        )
        return objVal_history

    def run(
        self,
        options: dict = None,
        solving_time_limit: float = 600,
        thread_limit: int = 1,
        solver: str = "gurobi",
    ):
        # Here, column generation
        # Subproblem : fixed destination
        self.runtime = 0
        self.lower_bound = 0
        self.upper_bound = 0
