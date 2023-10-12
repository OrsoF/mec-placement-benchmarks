from problem_classes.probleminstance import ProblemInstance
from gurobipy import GRB
import gurobipy as gp
import numpy as np

np.random.seed(0)
import math


class Solver:
    def __init__(self, problem_instance: ProblemInstance, **args) -> None:
        self.problem_instance = problem_instance
        self.runtime: float
        self.lower_bound: float
        self.upper_bound: float
        self.solver_name = ""

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
