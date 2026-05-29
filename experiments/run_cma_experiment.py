from instances.builder import make_star_instance, make_tree_instance

instance = make_star_instance(
    users_per_group=10, number_of_groups=3, neighbors_connection=2
)
# instance = make_tree_instance(7, 3)


from solvers.evolutionary.cma import Solver

solver = Solver(instance, 2)
solver.run()
