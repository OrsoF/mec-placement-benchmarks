"""Benchmark result schema and solver result extraction."""

RESULT_COLUMNS = (
    "instance",
    "solver",
    "users",
    "neighbors",
    "groups",
    "score",
    "runtime",
    "buildtime",
)

def result_row(args, name: str, users: int, groups: int, solver) -> dict:
    return {
        "instance": args.instance,
        "solver": name,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": solver.score,
        "runtime": solver.runtime,
        "buildtime": getattr(solver, "build_time", 0.0) or 0.0,
    }



def failed_row(args, name: str, users: int, groups: int, runtime: float) -> dict:
    return {
        "instance": args.instance,
        "solver": name,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": "",
        "runtime": runtime,
        "buildtime": "",
    }



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
