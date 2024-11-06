import os
import importlib
import pandas as pd
import pickle as pkl
import tqdm
from problem_classes.probleminstance import ProblemInstance, ToyInstance
from importlib.util import spec_from_file_location, module_from_spec
from utils.generic_solver import GenericSolver

INSTANCES_PATH = os.path.join(os.getcwd(), "instances")
SOLVER_PATH = os.path.join(os.getcwd(), "solvers")
CURRENT_FOLDER = os.getcwd()


def pickle_save(object_instance: object, file_name: str):
    with open(file_name, "wb") as handle:
        pkl.dump(object_instance, handle, protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, "rb") as handle:
        python_variable = pkl.load(handle)
    return python_variable


def folder_management():
    assert "gml_graphs" in os.listdir(
        CURRENT_FOLDER
    ), "There is not simple instances folder."
    assert os.listdir(
        os.path.join(CURRENT_FOLDER, "gml_graphs")
    ), "There is not simple instances."
    try:
        os.mkdir(os.path.join(CURRENT_FOLDER, "results"))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(CURRENT_FOLDER, "instances"))
    except FileExistsError:
        pass


def reset(verbose: bool = True):
    if verbose:
        print("Deleting instances...")
    instance_path = "instances"
    try:
        os.mkdir(os.path.join(os.getcwd(), instance_path))
    except FileExistsError:
        pass
    instance_list = os.listdir(os.path.join(os.getcwd(), instance_path))
    for current_instance in instance_list:
        current_instance_path = os.path.join(instance_path, current_instance)
        os.remove(current_instance_path)


def get_instance_list() -> list[ProblemInstance]:
    all_instances = []

    instance_path = "instances"
    instances_list = os.listdir(instance_path)
    for instance in instances_list:
        current_instance_path = os.path.join(instance_path, instance)
        all_instances.append(pickle_load(current_instance_path))

    return all_instances


def get_solver_list():
    solver_list = []

    solver_path = "solvers"
    solver_string_list = os.listdir(solver_path)
    for solver_str in solver_string_list:
        if solver_str != "__pycache__":
            solver_module = "{}.{}".format(solver_path, solver_str[:-3])
            current_solver = importlib.import_module(solver_module)
            solver_list.append(current_solver.Solver)
    return solver_list


def experience_index() -> int:
    """
    Returns the number of the current experience.
    """
    file_list = os.listdir(os.path.join(os.getcwd(), "results"))
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


def get_result_informations(solver_result) -> list:
    results = [0, 0, 0, 0]
    results[0] = get_info_from_results(solver_result, "Time: ")
    results[1] = get_info_from_results(solver_result, "Lower bound: ")
    results[2] = get_info_from_results(solver_result, "Upper bound: ")
    results[3] = get_info_from_results(solver_result, "Termination condition: ")
    return results


def get_info_from_results(result_type: str, info_string: str):
    """
    Get information about results from Pyomo result info string.
    """
    i = str(result_type).lower().find(info_string.lower()) + len(info_string)
    value = ""
    while str(result_type)[i] != "\n":
        value = value + str(result_type)[i]
        i += 1
    return value


def import_instance_from_file(instance_name: str) -> ProblemInstance:
    instance_name = (
        instance_name + ".pkl" if not instance_name.endswith(".pkl") else instance_name
    )
    instance_path = os.path.join(INSTANCES_PATH, instance_name)
    return pickle_load(instance_path)


def import_solver_from_file(
    solver_name: str, instance: ProblemInstance
) -> GenericSolver:
    solver_name = (
        solver_name + ".py" if not solver_name.endswith(".py") else solver_name
    )
    spec = spec_from_file_location(
        "__temp_module__", os.path.join(SOLVER_PATH, solver_name)
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver(instance)


def solve(
    instance_name: str,
    solver_name: str,
    repeat: int = 1,
    check_solution: bool = False,
    initial_variable=None,
    verbose:bool=False,
) -> tuple[ProblemInstance, GenericSolver]:
    """
    Allows to solve an instance by its name and by the solver name.
    """
    instance = import_instance_from_file(instance_name)
    solver = import_solver_from_file(solver_name, instance)

    if verbose:
        print('Start solving...')

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

    instance._set_arc_solution(solver.x_variable, solver.y_variable, solver.q_variable)
    if check_solution:
        instance.check_current_solution()
    return instance, solver
