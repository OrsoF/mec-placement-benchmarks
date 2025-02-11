from utils.build_instance_tree import make_tree_instance
from utils.data_management import solve

users = 7
neighbors = 3
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "rl_base"
make_tree_instance(number_of_users=users, neighbors_connection=neighbors)


instance, solver = solve(instance_name, solver_name)
print("RL runtime : {}".format(solver.runtime))
print("RL score : {}".format(solver.score))
# instance.print_current_solution(show_graph=False)
variable = (solver.x_variable, solver.y_variable, solver.q_variable)

instance, solver = solve(instance_name, "ILP_arc", initial_variable=variable)
print("Runtime with warm start : {}".format(solver.runtime))
print("Score with warm start : {}".format(solver.score))
# instance.print_current_solution(show_graph=False)

instance, solver = solve(instance_name, "ILP_arc")
print("Runtime with cold start : {}".format(solver.runtime))
print("Score with cold start : {}".format(solver.score))
# instance.print_current_solution(show_graph=False)
instance._set_arc_solution(solver.x_variable, solver.y_variable, solver.q_variable)
instance.check_current_solution()
# instance.print_current_solution()
