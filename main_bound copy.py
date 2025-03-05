import pandas as pd
import seaborn as sns
import numpy as np
from utils.build_instance_tree import make_tree_instance
from solvers.ILP_arc import Solver as ilp_arc_Solver
from solvers.LP_arc import Solver as lp_arc_Solver
from solvers.ILP_path import Solver as ilp_path_Solver
from solvers.LP_path_strong_latency import Solver as lp_path_strong_latency_Solver

import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Create instances with 50, 100, and 150 users
instances = {
    50: make_tree_instance(50, 1),
    100: make_tree_instance(100, 1),
    150: make_tree_instance(150, 1),
}

results = []


solvers = {
    "ILP": ilp_arc_Solver,
    "Relaxation": lp_arc_Solver,
    "ILP-Path": ilp_path_Solver,
    "Relaxation Enforced Latency": lp_path_strong_latency_Solver,
}

for num_users, instance in instances.items():
    print(num_users)
    for sol_name, Solver in solvers.items():
        sol = Solver(instance)
        sol.run()
        results.append(
            {
                "Solver": sol_name,
                "Users": num_users,
                "Objective Value": sol.score,
                "Time": sol.runtime,
            }
        )

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

common_params = {
    # 'linewidth': 2.5,  # Thicker lines
    "markersize": 8,  # Larger markers
    "ax": None,
}

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df[results_df["Solver"] != "ILP"],
    x="Users",
    y="Objective Value",
    hue="Solver",
    marker="o",
    linewidth=2.5,
    **common_params
)
sns.lineplot(
    data=results_df[results_df["Solver"] == "ILP"],
    x="Users",
    y="Objective Value",
    hue="Solver",
    marker="o",
    linewidth=3.5,
    **common_params
)  # Thicker line for Solver1
plt.ylabel("Objective value", fontsize=14)
plt.xlabel("Users", fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("objective_bound.pdf", bbox_inches="tight", dpi=300)
plt.show()
