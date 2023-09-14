import numpy as np
import gymnasium as gym


def train(agent, env, episodes, seed):
    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed)

        terminated, truncated = False, False
        step_idx = 0
        while not (terminated or truncated):
            action = agent.action(obs)
            obs, reward, terminated, truncated = env.step(action)

            # inform agent about the reward
            agent.update(obs, reward, terminated, truncated)
            step_idx += 1

        yield episode_idx


def evaluate(agent, env, episodes, seed, ):
    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed)

        terminated, truncated = False, False
        step_idx = 0
        while not (terminated or truncated):
            action = agent.action(obs)
            obs, reward, terminated, truncated = env.step(action)

            # inform agent about the reward
            agent.update(obs, reward, terminated, truncated)
            step_idx += 1

        yield episode_idx


class QLAgent:

    def __init__(self, learning_rate, discount, action_space, observation_space):
        self.qtable = np.zeros((observation_space.n, action_space.n))
        self.learning_rate = learning_rate
        self.discount = discount

    def action(self, observation):
        return np.argmax(self.qtable[observation, :])

    def update(self, obs, action, new_obs, reward, terminated, truncated):
        if terminated:
            target = reward
        else:
            target = reward + self.discount * np.max(self.qtable[new_obs, :])
        q_value = self.qtable[obs, action]
        self.qtable[obs, action] = (1 - self.learning_rate) * q_value + self.learning_rate * target


if __name__ == '__main__':
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=True
    )

    try:
        agent = QLAgent(env.action_space, env.observation_space)

        for episode_idx in train(agent, env, 100, seed=1889):
            pass
        a = 1
        print('test ql', a)
    finally:
        env.close()
