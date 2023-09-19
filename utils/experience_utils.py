import os

try:
    os.mkdir("results")
except FileExistsError:
    pass


def experience_index() -> int:
    """
    Returns the number of the current experience.
    """
    file_list = os.listdir(os.path.join(os.getcwd(), "results"))
    if len(file_list) == 0:
        return 0
    else:
        return max(int(file_name.split("_")[0]) for file_name in file_list) + 1
