import os
import importlib
import pandas as pd
import pickle as pkl
import tqdm
import pyomo as pyo
from core.problem_classes.problem_instance import ProblemInstance, ToyInstance
from importlib.util import spec_from_file_location, module_from_spec
from core.generic_solver import GenericSolver

ARTIFACTS_PATH = os.path.join(os.getcwd(), "artifacts")
INSTANCES_PATH = os.path.join(ARTIFACTS_PATH, "instances")
RESULTS_PATH = os.path.join(ARTIFACTS_PATH, "data")
SOLVER_PATH = os.path.join(os.getcwd(), "solvers")
CURRENT_FOLDER = os.getcwd()
SOLVER_FILE_ALIASES = {
    "ILP_arc": "ilp_arc",
    "ILP_arc_95": "ilp_arc_95_latency",
    "ILP_path": "ilp_path",
    "MILP_arc": "milp_arc",
    "MILP_path": "milp_path",
    "milp_arc_exact": "milp_arc_full",
    "ILP_arc_heuristic_placement": "ilp_arc_heuristic_placement",
    "ILP_arc_heuristic_warmstart": "ilp_arc_heuristic_warmstart",
    "ILP_arc_rl_warmstart": "ilp_arc_rl_warmstart",
    "LP_arc": "lp_arc",
    "LP_path": "lp_path",
    "LP_path_strong_latency": "lp_path_latency_tightened",
    "rl_A2C": "a2c",
    "rl_PPO": "ppo",
    "rl_PPO_options": "ppo_options",
}


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
    solver_list = []

    for root, _, files in os.walk(SOLVER_PATH):
        for solver_file in files:
            if solver_file == "__init__.py" or not solver_file.endswith(".py"):
                continue
            module_path = os.path.relpath(
                os.path.join(root, solver_file[:-3]), CURRENT_FOLDER
            )
            solver_module = module_path.replace(os.sep, ".")
            current_solver = importlib.import_module(solver_module)
            solver_list.append(current_solver.Solver)
    return solver_list


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


def get_result_informations(solver_result: str) -> list:
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
    solver_name = SOLVER_FILE_ALIASES.get(solver_name, solver_name)
    solver_file = solver_name + ".py" if not solver_name.endswith(".py") else solver_name
    direct_path = os.path.join(SOLVER_PATH, solver_file)
    if os.path.exists(direct_path):
        solver_path = direct_path
    else:
        matches = []
        for root, _, files in os.walk(SOLVER_PATH):
            if solver_file in files:
                matches.append(os.path.join(root, solver_file))
        assert matches, f"Solver file not found: {solver_file}"
        assert len(matches) == 1, f"Solver name is ambiguous: {solver_file}"
        solver_path = matches[0]

    spec = spec_from_file_location("__temp_module__", solver_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
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
        instance._set_arc_solution(
            solver.x_variable, solver.y_variable, solver.q_variable
        )
        instance.check_current_solution()
    return instance, solver


def print_constraint_variables(model) -> None:
    num_variables = len([v for v in model.component_data_objects(pyo.Var)])
    print("Number of variables:", num_variables)

    num_constraints = len([c for c in model.component_data_objects(pyo.Constraint)])
    print("Number of constraints:", num_constraints)
