from problem_classes.probleminstance import ProblemInstance


class InstanceSolver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.runtime: float
        self.lower_bound: float
        self.upper_bound: float
        self.solver_name = ""

    def run(self):
        # To be implemented in the real solver.
        pass
