from utils.make_tree_instance import make_tree_instance

make_tree_instance()

from utils.data_management import solve

instance, solver = solve("tree_7_2", "base_heuristic")
print(solver.runtime)
instance.print_current_solution(show_graph=False)
