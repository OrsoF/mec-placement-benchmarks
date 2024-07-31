from utils.data_management import folder_management
from utils.make_instances_from_gml import make_instance_from_gml
from utils.make_big_instances import make_big_instances
from utils.make_tree_instance import make_tree_instance

folder_management()

# Create basic instances (less than 50 nodes)
make_instance_from_gml()

# Create big custom instances (1000 nodes or more)
make_big_instances()

# Create simple tree instance
make_tree_instance()

