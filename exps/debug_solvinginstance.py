from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

users = 6
neighbors = 1
instance_name = "tree_{}_{}".format(users, neighbors)
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

from problem_classes.solvinginstance import SolvingInstance

sol = SolvingInstance(problem_instance=problem_instance)
sol.basic_placement_and_routing()
sol._maximize_quantities_greedy()

solver_name = "ILP_arc"
problem_instance, solver = solve(instance_name, solver_name, verbose=True)

sols = solver.x_variable, solver.y_variable, solver.q_variable

sol.reset_instance()
sol.set_variable_solution(*sols)

sol.render_solution()