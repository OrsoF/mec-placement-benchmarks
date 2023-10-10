from utils.data_management import (
    reset,
    get_instance_list,
    get_solver_list,
    experience_index,
    create_dataframe_from_dict,
    reorder_columns,
    folder_management,
)
from utils.make_instances_from_gml import make_instance_from_gml
from utils.make_big_instances import make_big_instances
import numpy as np
from problem_classes.probleminstance import ProblemInstance
from utils.instance_solver import InstanceSolver
from utils.check_network_graph import check_network_graph

# Delete all files, create folders
reset()
folder_management()

# Create basic instances (less than 50 nodes)
make_instance_from_gml()

# Create big custom instances (1000 nodes or more)
make_big_instances()

# Create a list of available instance
instance_list = get_instance_list()

# Create a list of available solvers
all_solvers = get_solver_list()


instance_list = instance_list[1:3]

runtimes = {}
objectives = {}

print("Solving all instances with all methods...")
for solver in all_solvers:
    for instance in instance_list:
        instance: ProblemInstance
        sol: InstanceSolver = solver(instance)

        print(sol.solver_name, instance.problem_name)

        check_network_graph(instance.network_graph.nx_graph)

        sol.run(options={"time": 3600, "threads": 1})
        result_name = (instance.problem_name, sol.solver_name)
        runtimes[result_name] = np.round(float(sol.runtime), 2)
        objectives[result_name] = np.round(float(sol.objective_value), 2)


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
