# This code is there to test all solvers on all instances of problem.

from utils.get_all_instances import all_instances
from utils.get_all_solvers import all_solvers

from problem_classes.probleminstance import ProblemInstance
from utils.generic_solver import GenericSolver
from utils.pandas_util import create_dataframe_from_dict, reorder_columns
from utils.check_network_graph import check_network_graph
from utils.experience_utils import experience_index
import numpy as np

number_of_services = 1

all_instances = all_instances[number_of_services : number_of_services + 1]

runtimes = {}
objectives = {}

for solver in all_solvers:
    for instance in all_instances:
        instance: ProblemInstance
        sol: GenericSolver = solver(instance)

        check_network_graph(instance.network_graph.nx_graph)

        sol.run(options={"time": 3600, "threads": 1})
        result_name = (instance.problem_name, sol.solver_name)
        runtimes[result_name] = np.round(float(sol.runtime), 2)
        objectives[result_name] = np.round(float(sol.objective_value), 2)

print("Success")
