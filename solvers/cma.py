from utils.generic_solver import GenericSolver
from problem_classes.probleminstance import ProblemInstance
from problem_classes.solvinginstance import SolvingInstance

import cma
import time
import numpy as np


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, max_pcc: int = 10):
        self.problem_instance = problem_instance
        self.nx_graph = self.problem_instance.network_graph.nx_graph
        self.max_pcc = max_pcc

        self.solver_name = "CMA-ES"

    def run(
        self,
    ):
        self.max_value = 0
        start_time = time.time()

        self.sol_ins = SolvingInstance(self.problem_instance)
        x0 = self.initial_value()

        es = cma.CMAEvolutionStrategy(x0, 2)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.objective(x) for x in solutions])
            es.logger.add()  # write data to disc to be plotted
            es.disp()

        # print("Meilleure solution CMA: {}".format(self.max_value))

        self.runtime = time.time() - start_time
        self.score = self.max_value

    def objective(self, value: np.ndarray):
        S = self.problem_instance.number_of_service
        D = self.problem_instance.total_number_of_demands

        placement = [int(elem) for elem in value[:S]]
        routing = [int(elem) for elem in value[S : S + D]]
        quantities = [int(elem) for elem in value[-D:]]
        it_works = self.sol_ins.test_solution(placement, routing, quantities)

        if it_works:
            score = sum(quantities)
            self.max_value = max(self.max_value, score)
            return -score
        else:
            return 0.0

    def initial_value(self, method: str = "heuristic") -> np.ndarray:
        if method == "basic":
            placement, routing, quantities = self.sol_ins.basic_placement_and_routing()
        elif method == "heuristic":
            from solvers.heuristic import Solver

            sol = Solver(self.problem_instance)
            sol.run()
            placement, routing, quantities = sol.placement, sol.routing, sol.quantities

        return np.array(placement + routing + quantities)
