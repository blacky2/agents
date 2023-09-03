import numpy as np
import gymnasium as gym


def train(agent, env, episodes, seed):

    # TODO logging and stuff

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed)

        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.action(obs)
            obs, reward, terminated, truncated = env.step(action)

            # inform agent about the reward


class QLAgent:

    def __init__(self, action_space, observation_space):
        self.qtable = np.zeros((observation_space.n, action_space.n))

    def action(self, observation):
        return np.argmax(self.qtable[observation, :])


if __name__ == '__main__':
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=True
    )

    try:
        agent = QLAgent(env.action_space, env.observation_space)

        train(agent, env, 100, seed=1889)
        a = 1
        print('test ql', a)
    finally:
        env.close()
