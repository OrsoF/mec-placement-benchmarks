from problem_classes.probleminstance import ProblemInstance


class InstanceSolver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.runtime: float = 0.0
        self.objective_value: float = 0.0
        self.solver_name = ""

    def run(self):
        # To be implemented in the real solver.
        pass
