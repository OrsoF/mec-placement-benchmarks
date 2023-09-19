print('Deleting instances...')

import os

instance_path = 'instances'
instance_list = os.listdir(os.path.join(os.getcwd(), instance_path))
for current_instance in instance_list:
    current_instance_path = os.path.join(instance_path, current_instance)
    os.remove(current_instance_path)