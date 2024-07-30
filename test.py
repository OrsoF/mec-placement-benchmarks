from utils.data_management import solve

instance, solver = solve('tree_0', 'ilp_arc')
print(solver.runtime)
instance.set_arc_solution(solver.x_variable, solver.y_variable, solver.q_variable)
instance.print_current_solution()
instance.check_current_solution()