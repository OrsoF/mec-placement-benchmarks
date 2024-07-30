from utils.data_management import solve

instance, solver = solve('tree_0', 'base_heuristic')
print(solver.runtime)
instance.print_current_solution(show_graph=False)