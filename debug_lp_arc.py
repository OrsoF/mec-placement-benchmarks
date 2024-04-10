from utils.data_management import get_instance_list

instance_list = get_instance_list()
problem_instance = instance_list[0]

from solvers.LP_arc import Solver

lp_arc = Solver(problem_instance)
lp_arc.run()
print(lp_arc.runtime)
