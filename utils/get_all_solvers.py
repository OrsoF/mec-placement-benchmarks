from utils.pickle_utils import pickle_load
import os
import importlib

all_solvers = []

solver_path = "solvers"
solver_list = os.listdir(solver_path)
for solver in solver_list:
    current_solver_path = os.path.join(solver_path, solver)

    if solver != "__pycache__":
        current_solver = importlib.import_module(solver_path + "." + solver[:-3])
        all_solvers.append(current_solver.Solver)
