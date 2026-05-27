import matplotlib.pyplot as plt
import pandas as pd
import argparse as ap
from datetime import datetime
import seaborn as sns

arg = ap.ArgumentParser()
arg.add_argument("--instance", type=str, required=True)
arg.add_argument("--neighbors", type=int, default=1)
arg.add_argument("--save", type=bool, default=True)
arg.add_argument("--buildtime", type=bool, default=False)
args = arg.parse_args()

instance_type = args.instance
assert instance_type in ["tree", "parallel"]
neighbors = args.neighbors
save = args.save

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = f"results_{current_date}.csv"
CSV_PATH = "results_02_03_25.csv"

df = pd.read_csv(CSV_PATH)

df = df[df["instance"] == instance_type]
df = df[df["neighbors"] == neighbors]
df = df.drop(["instance", "neighbors", "groups"], axis=1)

df = df.sort_values(by=["users"])
df = df[
    df["solver"].isin(
        [
            "PPO",
            "ilp_arc",
            # "ilp_arc_heuristic_placement",
            # "ilp_arc_heuristic_warmstart",
            "heuristic",
            # "ilp_arc_95",
            # "ilp_arc_rl_warm_start",
        ]
    )
]

print(df.head())

plt.figure(figsize=(10, 6))
for solver_name in set(df["solver"]):
    df_temp = df[df["solver"] == solver_name]
    plt.plot(df_temp["users"], df_temp["score"], marker="o", label=solver_name)

plt.xlabel("Graph size", fontsize=14)
plt.ylabel("Objective", fontsize=14)
plt.title(f"{instance_type} score vs graph size for each solver", fontsize=16)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()

RESULT_PATH = f"artifacts/data/{instance_type}_score.png"
if save:
    plt.savefig(RESULT_PATH)
plt.show()

plt.figure(figsize=(10, 6))
for solver_name in set(df["solver"]):
    df_temp = df[df["solver"] == solver_name]
    plt.plot(df_temp["users"], df_temp["runtime"], marker="o", label=solver_name)

    if df_temp["buildtime"].iloc[0] > 0.0 and args.buildtime:
        plt.plot(
            df_temp["users"],
            df_temp["buildtime"],
            marker="o",
            label=f"Build time {solver_name}",
        )

plt.xlabel("Graph size", fontsize=14)
plt.ylabel("runtime", fontsize=14)
plt.title(f"{instance_type} runtime vs graph size for each solver", fontsize=16)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()
RESULT_PATH = f"artifacts/data/{instance_type}_runtime.png"
if save:
    plt.savefig(RESULT_PATH)
plt.show()


plt.figure(figsize=(10, 6))
for solver_name in set(df["solver"]):
    df_temp = df[df["solver"] == solver_name]
    plt.plot(df_temp["runtime"], df_temp["score"], marker="o", label=solver_name)

plt.ylabel("Objective", fontsize=14)
plt.xlabel("Runtime", fontsize=14)
# plt.title(
#     f"{instance_type} Performance vs Objective for the differents methods.", fontsize=16
# )
plt.xscale("log")
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()

RESULT_PATH_PNG = f"artifacts/data/{instance_type}_performance.png"
RESULT_PATH_PDF = f"artifacts/data/{instance_type}_performance.pdf"

if save:
    plt.savefig(RESULT_PATH_PDF)
plt.show()

# Set seaborn style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Define color palette for better visibility
palette = sns.color_palette("colorblind", n_colors=len(set(df["solver"])))

# Plot each solver with a distinct color
for i, solver_name in enumerate(sorted(set(df["solver"]))):
    df_temp = df[df["solver"] == solver_name]
    if solver_name == "ilp_arc":
        solver_name = "ILP (optimal solution)"
    plt.plot(
        df_temp["runtime"],
        df_temp["score"],
        marker="o",
        label=solver_name,
        linewidth=2,
        markersize=6,
        color=palette[i],
    )

# Labels and scales
plt.ylabel("Objective", fontsize=14, fontweight="bold")
plt.xlabel("Runtime", fontsize=14, fontweight="bold")
plt.xscale("log")

# Improve legend placement
plt.legend(
    title="Method",
    fontsize=12,
    title_fontsize=13,
    loc="upper left",
    # bbox_to_anchor=(1, 1),
    frameon=True,
)

# Enable minor gridlines
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Use scientific notation if needed
plt.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

# Adjust layout
plt.tight_layout()

# Save high-quality version
RESULT_PATH_PNG = f"artifacts/data/{instance_type}_performance_sns.png"
RESULT_PATH_PDF = f"artifacts/data/{instance_type}_performance_sns.pdf"

if save:
    plt.savefig(RESULT_PATH_PDF, bbox_inches="tight", dpi=300)
plt.show()
