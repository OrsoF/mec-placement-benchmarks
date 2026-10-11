import importlib
import os
import pickle as pkl
from pathlib import Path

import pandas as pd

from mecbench.benchmarks.metrics import get_info_from_results as get_info_from_results
from mecbench.benchmarks.metrics import get_result_informations as get_result_informations
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem import ToyInstance as ToyInstance
from mecbench.solvers.base import GenericSolver
from mecbench.solvers.registry import SOLVER_FILE_ALIASES, SOLVER_MODULES

ARTIFACTS_PATH = os.path.join(os.getcwd(), "artifacts")
INSTANCES_PATH = os.path.join(ARTIFACTS_PATH, "instances")
RESULTS_PATH = os.path.join(ARTIFACTS_PATH, "data")
SOLVER_PATH = str(Path(__file__).resolve().parents[1] / "solvers")
CURRENT_FOLDER = os.getcwd()


def pickle_save(object_instance: object, file_name: str):
    with open(file_name, "wb") as handle:
        pkl.dump(object_instance, handle, protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(file_name: str):
    """Load any pickle file."""
    with open(file_name, "rb") as handle:
        python_variable = pkl.load(handle)
    return python_variable


def folder_management():
    topology_path = os.path.join(CURRENT_FOLDER, "data", "topologies")
    assert os.path.isdir(topology_path), "There is no topology data folder."
    assert os.listdir(topology_path), "There are no topology files."
    os.makedirs(INSTANCES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)


def reset(verbose: bool = True):
    if verbose:
        print("Deleting instances...")
    instance_path = INSTANCES_PATH
    os.makedirs(instance_path, exist_ok=True)
    instance_list = os.listdir(instance_path)
    for current_instance in instance_list:
        current_instance_path = os.path.join(instance_path, current_instance)
        os.remove(current_instance_path)


def get_instance_list() -> list[ProblemInstance]:
    all_instances = []

    instance_path = INSTANCES_PATH
    instances_list = os.listdir(instance_path)
    for instance in instances_list:
        current_instance_path = os.path.join(instance_path, instance)
        all_instances.append(pickle_load(current_instance_path))

    return all_instances


def get_solver_list():
    return [importlib.import_module(module_name).Solver for module_name in SOLVER_MODULES.values()]


def experience_index() -> int:
    """
    Returns the number of the current experience.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    file_list = os.listdir(RESULTS_PATH)
    if len(file_list) == 0:
        return 0
    else:
        return max(int(file_name.split("_")[0]) for file_name in file_list) + 1


def create_dataframe_from_dict(input_dict: dict):
    """
    Create a pandas.DataFrame from a dict {(row, col) : value}
    """
    # Extract unique row and column names
    rows = set()
    columns = set()
    for key in input_dict:
        row_name, column_name = key
        rows.add(row_name)
        columns.add(column_name)

    # Create an empty DataFrame with the extracted row and column names
    df = pd.DataFrame(index=rows, columns=columns)

    # Fill the DataFrame using the dictionary values
    for key, value in input_dict.items():
        row_name, column_name = key
        df.loc[row_name, column_name] = value

    return df


def reorder_columns(df1, df2):
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        if df1.shape[0] >= 1:
            # Get the column order from the first row of df1
            column_order = df1.iloc[0].index.tolist()

            # Reorder columns in df1
            df1_reordered = df1[column_order]

            # Reorder columns in df2 based on the column order of df1
            df2_reordered = df2[column_order]

            return df1_reordered, df2_reordered
        else:
            raise ValueError("The first DataFrame must have at least one row.")
    else:
        raise TypeError("Input arguments must be Pandas DataFrames.")


def import_instance_from_file(instance_name: str) -> ProblemInstance:
    instance_name = instance_name + ".pkl" if not instance_name.endswith(".pkl") else instance_name
    instance_path = os.path.join(INSTANCES_PATH, instance_name)
    return pickle_load(instance_path)


def import_solver_from_file(solver_name: str, instance: ProblemInstance) -> GenericSolver:
    solver_name = SOLVER_FILE_ALIASES.get(solver_name, solver_name)
    solver_name = solver_name[:-3] if solver_name.endswith(".py") else solver_name
    assert solver_name in SOLVER_MODULES, f"Unknown solver: {solver_name}"
    module = importlib.import_module(SOLVER_MODULES[solver_name])
    return module.Solver(instance)


def solve(
    instance_name: str,
    solver_name: str,
    repeat: int = 1,
    check_solution: bool = False,
    initial_variable=None,
    verbose: bool = False,
    instance: ProblemInstance = None,
    solver: GenericSolver = None,
) -> tuple[ProblemInstance, GenericSolver]:
    """
    Allows to solve an instance by its name and by the solver name.
    """
    if instance is None:
        instance = import_instance_from_file(instance_name)

    if solver is None:
        solver = import_solver_from_file(solver_name, instance)

    if verbose:
        print("Start solving...")

    if repeat == 1:
        if initial_variable is not None:
            solver.run(initial_variable)
        else:
            solver.run()
    else:
        runtime = 0.0
        for _ in range(repeat):
            if initial_variable is not None:
                solver.run(initial_variable)
            else:
                solver.run()
            runtime += solver.runtime
        solver.runtime = runtime / repeat

    if verbose:
        print(f"Score : {solver.score}")
        print(f"Runtime : {solver.runtime}")

    if check_solution:
        instance._set_arc_solution(solver.x_var, solver.y_var, solver.q_var)
        instance.check_current_solution()
    return instance, solver


def print_constraint_variables(model) -> None:
    import pyomo.environ as pyo

    num_variables = len([v for v in model.component_data_objects(pyo.Var)])
    print("Number of variables:", num_variables)

    num_constraints = len([c for c in model.component_data_objects(pyo.Constraint)])
    print("Number of constraints:", num_constraints)
