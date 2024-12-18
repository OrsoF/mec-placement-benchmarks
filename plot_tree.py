import os
import matplotlib.pyplot as plt

from utils.pickle import pickle_load, pickle_save

instance_type = "tree"

path_file = os.path.join("results", "dict_scores.pkl")
results_dict = pickle_load(path_file)

for item in results_dict.items():
    print(item)


results_dict = {
    key: results_dict[key]
    for key in results_dict.keys()
    if key[0] == instance_type and key[3] == 2
}

solvers_list = [key[1] for key in results_dict.keys()]
solvers_label = {solvers_list[index]: index for index in range(len(solvers_list))}
# colormap = plt.cm.get_cmap("tab10", len(solvers_list))

complexities = []

results_score = {}
results_runtime = {}

for key in results_dict.keys():
    value = results_dict[key]

    complexity = key[2]
    complexities.append(complexity)
    solver_name = key[1]

    score = value[0]
    runtime = value[1]

    if solver_name in results_score:
        results_score[solver_name].append(score)
        results_runtime[solver_name].append(runtime)
    else:
        results_score[solver_name] = [score]
        results_runtime[solver_name] = [runtime]

complexities = list(set(complexities))
complexities.sort()

for key in results_score:
    scores = results_score[key]
    plt.plot(complexities, scores, label=f"{key} score")

plt.ylabel("Score")
plt.xlabel("Complexity")
plt.legend()
plt.show()

for key in results_score:
    runtimes = results_runtime[key]
    plt.plot(complexities, runtimes, label=f"{key} runtime")

plt.ylabel("Runtime")
plt.xlabel("Complexity")
plt.yscale("log")
plt.legend()
plt.show()
