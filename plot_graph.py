from utils.build_instance_parallel import make_parallel_instance
from utils.build_instance_tree import make_tree_instance
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--instance",
    type=str,
    required=True,
    help="Type of instance to plot. Either 'parallel' or 'tree'",
)

args = argparser.parse_args()
instance_type = args.instance
assert instance_type in ["parallel", "tree"]

if instance_type == "parallel":
    instance = make_parallel_instance(6, 1, 4)
else:
    instance = make_tree_instance(5, 2)

G = instance.network_graph.nx_graph

import matplotlib.pyplot as plt
import networkx as nx

# Use spring layout for automatic positioning
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="skyblue",
    node_size=200,
    font_size=15,
    font_color="black",
    edge_color="gray",
)

# Show the plot
plt.title(f"{instance_type} shaped graph")
plt.show()
