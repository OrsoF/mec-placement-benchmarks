from utils.EnvSP import EnvSP
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve

users = 11
neighbors = 1
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
make_tree_instance(number_of_users=users, neighbors_connection=neighbors)
problem_instance, solver = solve(instance_name, solver_name)

env = EnvSP(
    problem_instance=problem_instance, sp_number=3, limit_step=100, verbose=False
)
env.reset()

for _ in range(100):
    action = env.action_space.sample()
    env.step(action)
