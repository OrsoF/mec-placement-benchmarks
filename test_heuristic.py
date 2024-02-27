from utils.data_management import (
    get_instance_list,
)

instance_list = get_instance_list()
instance = instance_list[50]

from solvers.base_heuristic import Solver

sol = Solver(instance, 1)
sol.run()
print(sol.upper_bound)
