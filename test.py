from utils.make_tree_instance import make_tree_instance

users = 4
neighbors = 3
instance_name = "tree_{}_{}".format(users, neighbors)

make_tree_instance(number_of_users=users, neighbors_connection=neighbors)

from utils.data_management import solve

instance, solver = solve(instance_name, "base_heuristic")
print(solver.runtime)
print(solver.score)
# instance.print_current_solution(show_graph=False)

variable = (solver.x_variable, solver.y_variable, solver.q_variable)


instance, solver = solve(instance_name, "ILP_arc", initial_variable=variable)
print("Runtime with warm start : {}".format(solver.runtime))
print(solver.score)
# instance.print_current_solution(show_graph=False)


instance, solver = solve(instance_name, "ILP_arc")
print("Runtime with cold start : {}".format(solver.runtime))
print(solver.score)
# instance.print_current_solution(show_graph=False)