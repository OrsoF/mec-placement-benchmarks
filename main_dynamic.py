import pandas as pd
import seaborn as sns
import numpy as np
from utils.build_instance_tree import make_tree_instance
from solvers.ILP_arc import Solver as ilp_arc_Solver
from solvers.LP_arc import Solver as lp_arc_Solver
from solvers.ILP_path import Solver as ilp_path_Solver
from solvers.LP_path_strong_latency import Solver as lp_path_strong_latency_Solver

import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 50

original_instance = make_tree_instance(users, 1)

from solvers.rl_PPO import Solver as RLSolver

sol = RLSolver(original_instance, max_step=5000, train_step=50000, evaluation_step=5000, method="PPO")
sol.run()

print(sol.runtime/10)
print(sol.score)