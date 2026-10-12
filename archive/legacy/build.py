from mecbench.compat.experiment_io import folder_management
from mecbench.instances.synthetic import make_tree_instance

folder_management()

# Create basic instances (less than 50 nodes)
# GML batch builders were removed from the simplified instances package.

# Create big custom instances (1000 nodes or more)
# Large instances are now created directly with mecbench.instances.synthetic.make_large_instance.

# Create simple tree instance
make_tree_instance()
