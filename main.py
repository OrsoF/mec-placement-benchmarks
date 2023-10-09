# Delete all files
import reset

# Create basic instances (less than 50 nodes)
import utils.make_instances_from_gml

# Create big custom instances (1000 nodes or more)
import utils.make_big_instances_from_gml

# Create a list of available instance
from utils.get_all_instances import all_instances

# Create a list of available solvers
from utils.get_all_solvers import all_solvers

from problem_classes.probleminstance import ProblemInstance
from utils.instance_solver import InstanceSolver
from utils.pandas_util import create_dataframe_from_dict, reorder_columns
from utils.check_a_network_graph import check_network_graph
from utils.experience_utils import experience_index
import numpy as np

all_instances = all_instances[1:3]

runtimes = {}
objectives = {}

print("Solving all instances with all methods...")
for solver in all_solvers:
    for instance in all_instances:
        instance: ProblemInstance
        sol: InstanceSolver = solver(instance)

        print(sol.solver_name, instance.problem_name)

        check_network_graph(instance.network_graph.nx_graph)

        sol.run(options={"time": 3600, "threads": 1})
        result_name = (instance.problem_name, sol.solver_name)
        runtimes[result_name] = np.round(float(sol.runtime), 2)
        objectives[result_name] = np.round(float(sol.objective_value), 2)

import os

try:
    os.mkdir("results")
except FileExistsError:
    pass

index = experience_index()

runtimes_dataframe = create_dataframe_from_dict(runtimes)
objectives_dataframe = create_dataframe_from_dict(objectives)

objectives_dataframe, runtimes_dataframe = reorder_columns(
    objectives_dataframe, runtimes_dataframe
)

print(runtimes_dataframe)
runtimes_dataframe.to_excel("results/{}_runtimes.xlsx".format(index))
print()
print(objectives_dataframe)
objectives_dataframe.to_excel("results/{}_objectives.xlsx".format(index))
