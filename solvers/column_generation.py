from problem_classes.probleminstance import ProblemInstance
from gurobipy import GRB
import gurobipy as gp
import numpy as np
from solvers.base_heuristic import Solver as base_heuristic
from utils.service_placement import greedy_service_placement
from solvers.LP_arc import Solver as lp_arc_Solver

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
        """
        Generate an initial solution to the problem.
        """
        base_heuristic_solver = base_heuristic(self.problem_instance)
        base_heuristic_solver.run()
        service_node_matrix: np.ndarray = base_heuristic_solver.y_variable  # y
        demand_arc_transit: np.ndarray = base_heuristic_solver.x_variable  # x
        quantity_matrix: np.ndarray = base_heuristic_solver.q_variable  # p
        return service_node_matrix, demand_arc_transit, quantity_matrix

    def generate_initial_patterns(self):
        """
        We generate a part of the variables : some service placement.
        """
        # Variables : x, y, p, P, L. P c'est la quantité par arc, L la latence par arc.
        # x_d^a : la demande d utilise l'arc a
        # y_S^e : le service S utilise le noeud e
        # p_d : la quantité utilisée par la demande d

        # Générer un placement est un problème à part entière, on va générer un seul placement de service. Question : est-ce qu'on peut laisser des variables non fixées ?
        # On va essayer en générant juste l'attribution greedy des serveurs.
        service_placement = greedy_service_placement(
            self.available_resources, self.needed_resources
        )
        return [service_placement]

    def define_master_problem(self, pattern):
        """
        For the given instance, we solve the problem for the given pattern/service placement.
        """
        lp_arc_solver = lp_arc_Solver(self.problem_instance)
        lp_arc_solver._build_whole_problem()
        lp_arc_solver.model.y.fixed = True
        lp_arc_solver.model.y.value = np.array(pattern)
        return lp_arc_solver.model

    def define_subproblem(self, duals):
        """
        Now, we price the pattern. Most difficult part.
        """
        pass

    def col_gen(self):
        # Generate initial solution
        initial_solution = self.generate_initial_solution()
        objective_value_history = []

        while True:
            master_problem = self.master_problem(ins_, initial_solution)
            master_problem.optimize()
            objective_value_history.append(master_problem.objVal)
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
        return objective_value_history

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
