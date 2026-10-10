from mecbench.solvers.base import GenericSolver
from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.exact.ilp_arc import Solver as ILPArcSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "ILP_arc_heuristic_warmstart"

    def run(self):
        solver = ILPArcSolver(self.problem_instance)
        solver.run(heuristic_solution=True)
        self._set_output_variables(solver.x_var, solver.y_var, solver.q_var)
        self.score = solver.score
        self.runtime = solver.runtime
        self.build_time = solver.build_time
        return self
