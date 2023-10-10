import os

current_folder = os.getcwd()


def folder_management():
    assert "instances" in os.listdir(
        current_folder
    ), "There is not simple instances folder."
    assert os.listdir(
        os.path.join(current_folder, "instances")
    ), "There is not simple instances."
    try:
        os.mkdir(os.path.join(current_folder, "results"))
    except FileExistsError:
        pass
