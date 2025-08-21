from utils.build_instance_parallel import build_star_instance

instance = build_star_instance(5, 2, 3)

from utils.gym_instance import SPPEnv

env = SPPEnv(instance, 500, 5)
for _ in range(5000):
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)
    if terminated:
        env.reset()
        print("reset")
