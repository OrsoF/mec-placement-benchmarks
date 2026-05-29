from data_io.experiment_io import (
    reset,
    get_instance_list,
    get_solver_list,
    experience_index,
    create_dataframe_from_dict,
    folder_management,
)
import numpy as np
from core.problem_classes.problem_instance import ProblemInstance
from core.generic_solver import GenericSolver
import multiprocessing


def check_network_graph(nx_graph):
    """Legacy placeholder for a removed graph validation helper."""
    return nx_graph is not None

# Delete all files, create folders
reset()
folder_management()

# Instance batch generation was removed from the simplified instances package.

# Create a list of available instance
instance_list = get_instance_list()

# Create a list of available solvers
all_solvers = get_solver_list()

experience_number = experience_index()
instance_list = instance_list[1:2]

runtimes = {}
lower_bounds = {}
upper_bounds = {}

solving_time_limit = 60

print("Solving all instances with all methods...")
for solver in all_solvers:
    for instance in instance_list:
        instance: ProblemInstance
        sol: GenericSolver = solver(instance)

        print(sol.solver_name, instance.problem_name)

        check_network_graph(instance.network_graph.nx_graph)

        sol.run(solving_time_limit=solving_time_limit)

        result_name = (instance.problem_name, sol.solver_name)
        runtimes[result_name] = np.round(float(sol.runtime), 2)
        lower_bounds[result_name] = np.round(float(sol.lower_bound), 2)
        upper_bounds[result_name] = np.round(float(sol.upper_bound), 2)

        runtimes_df = create_dataframe_from_dict(runtimes)
        runtimes_df.to_excel("artifacts/data/{}_runtimes.xlsx".format(experience_number))
        lower_bounds_df = create_dataframe_from_dict(lower_bounds)
        lower_bounds_df.to_excel(
            "artifacts/data/{}_lower_bounds.xlsx".format(experience_number)
        )
        upper_bounds_df = create_dataframe_from_dict(upper_bounds)
        upper_bounds_df.to_excel(
            "artifacts/data/{}_upper_bounds.xlsx".format(experience_number)
        )


runtimes_df = create_dataframe_from_dict(runtimes)
runtimes_df.to_excel("artifacts/data/{}_runtimes.xlsx".format(experience_number))
print()
print("Runtimes :")
print(runtimes_df)
print()

lower_bounds_df = create_dataframe_from_dict(lower_bounds)
lower_bounds_df.to_excel("artifacts/data/{}_lower_bounds.xlsx".format(experience_number))
print("Lower bounds :")
print(lower_bounds_df)
print()

upper_bounds_df = create_dataframe_from_dict(upper_bounds)
upper_bounds_df.to_excel("artifacts/data/{}_upper_bounds.xlsx".format(experience_number))
print("Upper bounds :")
print(upper_bounds_df)
print()
