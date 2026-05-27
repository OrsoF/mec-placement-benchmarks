from data_io.data_management import folder_management
from instances.build_gml import make_instance_from_gml
from instances.build_large import make_big_instances
from instances.build_tree import make_tree_instance

folder_management()

# Create basic instances (less than 50 nodes)
# make_instance_from_gml()

# Create big custom instances (1000 nodes or more)
# make_big_instances()

# Create simple tree instance
make_tree_instance()
