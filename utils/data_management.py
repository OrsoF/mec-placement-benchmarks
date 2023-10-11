import os
from utils.pickle_utils import pickle_load
import importlib
import pandas as pd
import pickle as pkl


def pickle_save(object_instance: object, file_name: str):
    with open(file_name, "wb") as handle:
        pkl.dump(object_instance, handle, protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, "rb") as handle:
        python_variable = pkl.load(handle)
    return python_variable


current_folder = os.getcwd()


def folder_management():
    assert "gml_graphs" in os.listdir(
        current_folder
    ), "There is not simple instances folder."
    assert os.listdir(
        os.path.join(current_folder, "gml_graphs")
    ), "There is not simple instances."
    try:
        os.mkdir(os.path.join(current_folder, "results"))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(current_folder, "instances"))
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


def get_instance_list() -> list:
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


def get_result_informations(solver_result) -> tuple:
    tuple_of_results = [0, 0, 0, 0]
    tuple_of_results[0] = get_info_from_results(solver_result, "Time: ")
    tuple_of_results[1] = get_info_from_results(solver_result, "Lower bound: ")
    tuple_of_results[2] = get_info_from_results(solver_result, "Upper bound: ")
    tuple_of_results[3] = get_info_from_results(
        solver_result, "Termination condition: "
    )
    return tuple_of_results


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
