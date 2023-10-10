import os

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
