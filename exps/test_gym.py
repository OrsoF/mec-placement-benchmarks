from utils.make_parallel_tree import make_parallel_tree_instance

instance = make_parallel_tree_instance(5, 2, 3)

from utils.gym_instance import SPPEnv

env = SPPEnv(instance, 500, 5)
for _ in range(5000):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    if terminated:
        env.reset()
        print("reset")
