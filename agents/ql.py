import numpy as np
import gymnasium as gym

from queue import Queue
from threading import Thread

import plotly
import plotly.subplots
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback


def dash_worker(msg_queue: Queue):
    fig = go.Figure()
    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('Train RL'),
        dcc.Graph(id='rl-graph'),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
        dcc.Graph(figure=fig)
    ])

    data = {
        'episode': [],
        'total_reward': [],
        'steps': [],
    }

    @callback(Output('rl-graph', 'figure'), Input('interval-component', 'n_intervals'))
    def update_rl_graph(n):
        episode, values = msg_queue.get()

        if not data['episode'] or data['episode'][-1] < episode:
            data['episode'].append(episode)
            data['total_reward'].append(values['total_reward'])
            data['steps'].append(values['steps'])
        elif data['episode'][-1] == episode:
            data['total_reward'][-1] = values['total_reward']
            data['steps'][-1] = values['steps']
        else:
            print(f'skipped update for episode {episode}')

        fig = plotly.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
        fig.append_trace({
            'x': data['episode'],
            'y': data['total_reward'],
            'name': 'Total Reward',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
        fig.append_trace({
            'x': data['episode'],
            'y': data['steps'],
            'name': 'Steps',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 2, 1)

        return fig

    app.run_server(debug=True, use_reloader=False)


def train(agent, env, episodes, seed):
    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed)

        terminated, truncated = False, False
        step_idx = 0
        while not (terminated or truncated):
            action = agent.explore(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)

            # inform agent about the reward
            agent.inform(obs, action, new_obs, reward, terminated, truncated)
            obs = new_obs
            step_idx += 1

        yield episode_idx


def evaluate(agent, env, episodes, seed=None):
    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed)

        terminated, truncated = False, False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action = agent.action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        yield (
            episode_idx+1,
            {'total_reward': total_reward, 'steps': steps}
        )


def episode_mean(episode_iter):
    mean_values = {}
    for episode, values in episode_iter:
        if episode == 1:
            mean_values.update(values)
        else:
            for key, value in values.items():
                mean_values[key] = (mean_values[key] * (episode - 1) + value) / episode
        yield episode, mean_values


class QLAgent:

    def __init__(
        self,
        learning_rate,
        discount,
        greedyness,
        action_space,
        observation_space,
        seed=None
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qtable = np.zeros((observation_space.n, action_space.n))
        self.learning_rate = learning_rate
        self.discount = discount
        self.greedyness = greedyness
        self.rand = np.random.default_rng(seed)

    def action(self, observation):
        return np.argmax(self.qtable[observation, :])

    def explore(self, observation):
        p = self.rand.uniform(0, 1)
        if p < self.greedyness:
            return self.action_space.sample()
        return self.action(observation)

    def inform(self, obs, action, new_obs, reward, terminated, truncated):
        if terminated:
            target = reward
        else:
            target = reward + self.discount * np.max(self.qtable[new_obs, :])
        q_value = self.qtable[obs, action]
        self.qtable[obs, action] = (1 - self.learning_rate) * q_value + self.learning_rate * target


if __name__ == '__main__':
    print('after dash')
    msg_queue = Queue(maxsize=0)
    dash_worker = Thread(target=dash_worker, args=(msg_queue,))
    dash_worker.daemon = True
    dash_worker.start()

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=True
    )

    try:
        agent = QLAgent(
            learning_rate=0.9,
            discount=0.99,
            greedyness=0.4,
            action_space=env.action_space,
            observation_space=env.observation_space
        )

        for episode_idx in train(agent, env, 10000, seed=1889):
            if episode_idx % 10 == 0:
                # mean steps
                for eval_episode, values in episode_mean(evaluate(agent, env, 10)):
                    msg_queue.put((episode_idx+1, values))

    finally:
        env.close()

    input()
