from utils.build_instance_tree import make_tree_instance
from utils.data_management import solve, import_instance_from_file, import_solver_from_file
import os
from solvers.rl_A2C import Solver

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_size = 10

# Small instance
users = 4
neighbors = 1

# Small instance
users = 10
neighbors = 2

# # Average instance
users = 20
neighbors = 4

# # Large instance
# users = 30
# neighbors = 6

instance_name = "tree_{}_{}".format(users, neighbors)
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
training_instances_database = [make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors, randomize=True
) for _ in range(train_size)]
training_instances_database = None

solver_name = "rl_A2C"

instance = import_instance_from_file(instance_name)
solver = Solver(problem_instance, training_instances_database, 2500, 50000, 25000)
solver.run()
print(f"Score : {solver.score}")
print(f"Runtime : {solver.runtime}")