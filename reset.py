# This code delete all instances of problem.

print("Deleting instances...")

import os

instance_path = "instances"
try:
    os.mkdir(os.path.join(os.getcwd(), instance_path))
except FileExistsError:
    pass
instance_list = os.listdir(os.path.join(os.getcwd(), instance_path))
for current_instance in instance_list:
    current_instance_path = os.path.join(instance_path, current_instance)
    os.remove(current_instance_path)
