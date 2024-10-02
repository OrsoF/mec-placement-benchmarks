from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve

users = 8
neighbors = 2
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
make_tree_instance(number_of_users=users, neighbors_connection=neighbors)
problem_instance, solver = solve(instance_name, solver_name)

print("Heuristique ok")
env = SPPGym(problem_instance=problem_instance)
env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    # print(obs)

instance, solver = solve(instance_name, "ILP_arc")
print("Runtime: {}".format(solver.runtime))
print("Score: {}".format(solver.score))
