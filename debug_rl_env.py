from utils.SPPGym import SPPGym
from utils.make_tree_instance import make_tree_instance
from utils.data_management import solve
from stable_baselines3 import A2C
import os

users = 8
neighbors = 4
instance_name = "tree_{}_{}".format(users, neighbors)
solver_name = "base_heuristic"
problem_instance = make_tree_instance(
    number_of_users=users, neighbors_connection=neighbors
)
# problem_instance, solver = solve(instance_name, solver_name)
# print("Heuristique ok")

save_path = os.path.join("results", "agent_a2c_" + instance_name)

env = SPPGym(problem_instance=problem_instance)
env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    # print(obs)
    if terminated:
        env.reset()

env.reset()
model = A2C("MlpPolicy", env, verbose=0, seed=0)

# if os.path.exists(save_path):
#     model.load(save_path)
model.learn(total_timesteps=50000, progress_bar=True)
# model.save(save_path)

print("Meilleure solution RL: {}".format(env.max_q_reached))

# print("Starting linear solving.")

# x_var, y_var, q_var = env.observation_to_variables(env.state)
# variable = (x_var, y_var, q_var)

# instance, solver = solve(instance_name, "ILP_arc", initial_variable=variable)
# print("Runtime with warm start : {}".format(solver.runtime))
# print("Score with warm start : {}".format(solver.score))

# instance, solver = solve(instance_name, "ILP_arc")
# print("Runtime with cold start : {}".format(solver.runtime))
# print("Score with cold start : {}".format(solver.score))
