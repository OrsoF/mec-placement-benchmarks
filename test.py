from utils.data_management import solve

instance, solver = solve('tree_0', 'ilp_arc')
print(solver.runtime)
print(solver.score)
instance.print_current_solution(show_graph=False)