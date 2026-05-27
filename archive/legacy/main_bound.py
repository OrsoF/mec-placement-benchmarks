import pandas as pd
import seaborn as sns
import numpy as np
from instances.tree_builder import make_tree_instance
from solvers.exact.ilp_arc import Solver as ilp_arc_Solver
from solvers.relaxations.lp_arc import Solver as lp_arc_Solver
from solvers.exact.ilp_path import Solver as ilp_path_Solver
from solvers.relaxations.lp_path_strong_latency import Solver as lp_path_strong_latency_Solver

import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Create instances with 50, 100, and 150 users
le_range = range(20, 80, 20)

instances = {nb: make_tree_instance(nb, 1) for nb in le_range}

results = []


solvers = {
    "ILP": ilp_arc_Solver,
    "Relaxation": lp_arc_Solver,
    # "ILP-Path": ilp_path_Solver,
    "ILP-Path": ilp_arc_Solver,
    "Relaxation Enforced Latency": lp_path_strong_latency_Solver,
}

for num_users, instance in instances.items():
    print(num_users)
    for sol_name, Solver in solvers.items():
        print(sol_name)
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

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage{bm}",
                r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}",
            ]
        ),
        "font.family": "serif",
        "font.size": 10,
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

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
