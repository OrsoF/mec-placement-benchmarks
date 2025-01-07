from utils.make_parallel_tree import make_parallel_tree_instance
from utils.make_tree_instance import make_tree_instance

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
    node_color='skyblue',
    node_size=200,
    font_size=15,
    font_color='black',
    edge_color='gray',
)

# Show the plot
plt.title("Parallel trees shaped graph")
plt.show()
