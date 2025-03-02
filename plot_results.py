import matplotlib.pyplot as plt
import pandas as pd
import argparse as ap
from datetime import datetime

arg = ap.ArgumentParser()
arg.add_argument("--instance", type=str, required=True)
arg.add_argument("--neighbors", type=int, default=1)
arg.add_argument("--save", type=bool, default=False)
arg.add_argument("--buildtime", type=bool, default=False)
args = arg.parse_args()

instance_type = args.instance
assert instance_type in ["tree", "parallel"]
neighbors = args.neighbors
save = args.save

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = f"results_{current_date}.csv"
# CSV_PATH = "results_12_02_25.csv"

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
            "ilp_arc_heuristic_placement",
            "ilp_arc_heuristic_warmstart",
            "heuristic",
            "ilp_arc_95",
            "ilp_arc_rl_warm_start",
        ]
    )
]

plt.figure(figsize=(10, 6))
for solver_name in set(df["solver"]):
    df_temp = df[df["solver"] == solver_name]
    plt.plot(df_temp["users"], df_temp["score"], marker="o", label=solver_name)

plt.xlabel("Graph size", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.title(f"{instance_type} score vs graph size for each solver", fontsize=16)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()

RESULT_PATH = f"results/{instance_type}_score.png"
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
RESULT_PATH = f"results/{instance_type}_runtime.png"
if save:
    plt.savefig(RESULT_PATH)
plt.show()


plt.figure(figsize=(10, 6))
for solver_name in set(df["solver"]):
    df_temp = df[df["solver"] == solver_name]
    plt.plot(df_temp["score"], df_temp["runtime"], marker="o", label=solver_name)

plt.xlabel("Score", fontsize=14)
plt.ylabel("Runtime", fontsize=14)
plt.title(
    f"{instance_type} Performance vs Score for the differents methods.", fontsize=16
)
plt.legend(title="Method", fontsize=10)
plt.grid(True)
plt.tight_layout()

RESULT_PATH = f"results/{instance_type}_performance.png"

if save:
    plt.savefig(RESULT_PATH)
plt.show()
