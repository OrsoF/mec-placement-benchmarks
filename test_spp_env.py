from instances.tree_builder import make_tree_instance
from rl.spp_env import SPPEnv


def main() -> None:
    instance = make_tree_instance(number_of_users=10, neighbors_connection=0)
    env = SPPEnv(
        instance,
        max_step=1,
        max_path_candidates=2,
        warm_start_heuristic=False,
        render_mode="human",
    )

    env.reset()
    action = env.action_space.sample()
    env.step(action)
    env.render()


if __name__ == "__main__":
    main()
