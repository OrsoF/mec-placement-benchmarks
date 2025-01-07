import os
import matplotlib.pyplot as plt
from utils.pickle import pickle_load, pickle_save
import pandas as pd

instance_type = "tree"
path_file = os.path.join("results", "dict_scores.pkl")
results_dict = pickle_load(path_file)

# results_dict = {
#     key: results_dict[key]
#     for key in results_dict.keys()
#     if key[0] == instance_type and key[3] == 2
# }

df = pd.DataFrame(
    [
        (key[0], key[1], key[2], key[3], value[0], value[1])
        for key, value in results_dict.items()
    ],
    columns=["Type", "Method", "Node", "Neighbors", "Score", "Time"],
)
df = df[df["Type"] == "tree"]
df = df[df["Neighbors"] == 2]

grouped = df.groupby("Method")
plt.figure(figsize=(10, 6))

for method, group in grouped:
    group = group.sort_values(by="Node")
    plt.plot(group["Node"], group["Score"], marker="o", label=method)

plt.xlabel("Node", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.title("Score vs Node for Each Method", fontsize=16)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for method, group in grouped:
    group = group.sort_values(by="Node")
    plt.plot(group["Node"], group["Time"], marker="o", label=method)

plt.xlabel("Node", fontsize=14)
plt.ylabel("Time", fontsize=14)
plt.title("Time vs Node for Each Method", fontsize=16)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
