from core.generic_solver import GenericSolver
from core.problem_classes.problem_instance import ProblemInstance
from solvers.exact.ilp_arc import Solver as ILPArcSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "ILP_arc_95_latency"

    def run(self):
        solver = ILPArcSolver(self.problem_instance)
        solver.run(options={"MIPGap": 0.05})
        self._set_output_variables(solver.x_var, solver.y_var, solver.q_var)
        self.score = solver.score
        self.runtime = solver.runtime
        self.build_time = solver.build_time
        return self
