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
