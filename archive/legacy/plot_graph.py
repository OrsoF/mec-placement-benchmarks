from instances.parallel_builder import build_star_instance
from instances.tree_builder import make_tree_instance
import argparse
import matplotlib.pyplot as plt
import networkx as nx

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

users = 6
edge = 2
cloud = 1
neighbors = 1

if instance_type == "parallel":
    instance = build_star_instance(users // edge, neighbors, edge)
else:
    instance = make_tree_instance(users, neighbors)

G = instance.network_graph.nx_graph


if instance_type == "tree":
    assert (
        len(G.nodes) == users + 1 + cloud
    ), f"Expected {users + 1 + cloud} nodes, got {len(G.nodes)}"

    labels = [f"User {i+1}" for i in range(users)] + ["Edge"] * 1 + ["Cloud"] * cloud
    node_labels = {i: labels[i] for i in range(users + 1 + cloud)}
    nx.relabel_nodes(
        G,
        node_labels,
        copy=False,
    )

    def tree_layout(G, users):
        pos = {f"User {user + 1}": (user, 0) for user in range(users)}
        pos["Edge"] = (sum(range(users)) / users, 1)
        pos["Cloud"] = (pos["Edge"][0], 2)
        return pos

    pos = tree_layout(G, users)

else:
    pass

print(pos)


# pos = nx.spring_layout(G)

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
RESULT_PATH = f"artifacts/data/{instance_type}_layout.png"
plt.savefig(RESULT_PATH)
plt.show()
