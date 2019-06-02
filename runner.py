
import numpy as np

from envs.wrappers import NormalizedBoxEnv
from gym.envs.mujoco import HalfCheetahEnv
from envs.ant_dir import AntDirEnv
from envs.ant_goal import AntGoalEnv
from envs.point_mass import PointEnv


def main():

    task_params = dict(
                n_tasks=2,  # 20 works pretty well
                forward_backward=True,
                tasks_id=1,
                randomize_tasks=True,
                low_gear=False,
            )
    env = NormalizedBoxEnv(
        AntDirEnv(
            n_tasks=task_params['n_tasks'],
            forward_backward=task_params['forward_backward'],
            tasks_id=task_params['tasks_id'],
            use_low_gear_ratio=task_params['low_gear']
        )
    )



    # task_params = dict(
    #     n_tasks=20,  # 20 works pretty well
    # )
    # env = NormalizedBoxEnv(
    #     AntGoalEnv(
    #         n_tasks=task_params['n_tasks']
    #     )
    # )
    print(env.tasks)

    tasks = env.get_all_task_idx()
    # print(tasks)

    # for i in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())
    #     print(env.action_space.sample())
    # env.close()

    for i_episode in range(10):
        # env.reset_task(np.random.randint(2))
        env.reset()
        print("Goal: ", env._goal)
        for t in range(100):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print("Reward: ", reward)


            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))


if __name__ == '__main__':
    main()
