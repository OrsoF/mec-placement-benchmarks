from data_io.experiment_io import folder_management
from instances.gml_builder import make_instance_from_gml
from instances.large_builder import make_big_instances
from instances.tree_builder import make_tree_instance

folder_management()

# Create basic instances (less than 50 nodes)
# make_instance_from_gml()

# Create big custom instances (1000 nodes or more)
# make_big_instances()

# Create simple tree instance
make_tree_instance()
