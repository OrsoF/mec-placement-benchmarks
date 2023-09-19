from utils.pickle_utils import pickle_load
import os

all_instances = []

instance_path = "instances"
instances_list = os.listdir(instance_path)
for instance in instances_list:
    current_instance_path = os.path.join(instance_path, instance)
    all_instances.append(pickle_load(current_instance_path))
