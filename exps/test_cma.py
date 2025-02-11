from utils.build_instance_parallel import make_parallel_tree_instance
from utils.build_instance_tree import make_tree_instance

instance = make_parallel_tree_instance(
    number_of_users_per_group=10, number_of_groups=3, neighbors_connection=2
)
# instance = make_tree_instance(7, 3)


from solvers.cma import Solver

solver = Solver(instance, 2)
solver.run()
