import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.concat([pd.read_csv("final.csv")], ignore_index=True)
df = df[df["instance"] == "tree"]
solvers_to_remove = [
    "ilp_arc_95",
    "ilp_arc_heuristic_placement",
    "ilp_arc_heuristic_warm_start",
    "ilp_arc_rl_warm_start",
]
df = df[~df["solver"].isin(solvers_to_remove)]
df.reset_index(drop=True, inplace=True)

df = df[df["users"] <= 220]

df["solver"] = df["solver"].replace("ilp_arc", "ILP")
df["solver"] = df["solver"].replace("heuristic", "Heuristic")

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

fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Common parameters for the plots
common_params = {
    "data": df,
    "x": "users",
    "hue": "solver",
    "marker": "o",
    "linewidth": 2.5,  # Thicker lines
    "markersize": 8,  # Larger markers
    "ax": None,
}

fontsize_title = 22
fontsize_labels = 20
fontsize_legend = 18
title_fontsize = 20

# Plot runtime vs users
common_params["ax"] = axes[1]
sns.lineplot(y="runtime", **common_params)
axes[1].set_title("Solver Runtime vs. Graph Size", fontsize=fontsize_title)
axes[1].set_xlabel("Graph size", fontsize=fontsize_labels)
axes[1].set_ylabel("Runtime (s)", fontsize=fontsize_labels)
axes[1].set_yscale("log")
axes[1].legend(title="Solver", fontsize=fontsize_legend, title_fontsize=title_fontsize)
axes[1].grid(True)

# Plot score vs users
common_params["ax"] = axes[0]
sns.lineplot(y="score", **common_params)
axes[0].set_title("Solver Objective vs. Graph Size", fontsize=fontsize_title)
axes[0].set_xlabel("Graph Size", fontsize=fontsize_labels)
axes[0].set_ylabel("Objective Value", fontsize=fontsize_labels)
axes[0].legend(title="Solver", fontsize=fontsize_legend, title_fontsize=title_fontsize)
axes[0].grid(True)
axes[0].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("final_tree.pdf", bbox_inches="tight", dpi=300)
plt.show()
