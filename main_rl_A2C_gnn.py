from utils.make_tree_instance import make_tree_instance
from utils.data_management import (
    solve,
    import_instance_from_file,
    import_solver_from_file,
)
from solvers.rl_A2C_GNN import Solver
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
users = 5
neighbors = 1

instance_name = "tree_{}_{}".format(users, neighbors)
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)

solver = Solver(
    problem_instance,
    training_instances_database=None,
    max_step=200,
    train_step=2000,
    evaluation_step=2000,
)
solver.run()
print(f"Score : {solver.score}")
print(f"Runtime : {solver.runtime}")
