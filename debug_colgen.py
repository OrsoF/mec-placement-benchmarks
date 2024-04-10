from utils.data_management import get_instance_list

instance_list = get_instance_list()
problem_instance = instance_list[0]

from solvers.column_generation import Solver

colgen = Solver(problem_instance)
colgen.run()
