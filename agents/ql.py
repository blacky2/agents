import numpy as np
import gymnasium as gym

import numbers
from queue import Queue
from threading import Thread
from collections import namedtuple

import plotly
import plotly.subplots
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback


def dash_worker(msg_queue: Queue):
    # fig = go.Figure()
    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('Train RL'),
        dcc.Graph(id='rl-graph'),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
        # dcc.Graph(figure=fig)
    ])

    train_data = {
        'episode': [],
        'total_reward': [],
        'steps': [],
    }

    eval_data = {
        'episode': [],
        'total_reward': [],
        'steps': [],
    }

    @callback(Output('rl-graph', 'figure'), Input('interval-component', 'n_intervals'))
    def update_rl_graph(n):
        episode, value_type, values = msg_queue.get()

        data = eval_data if value_type == 'eval' else train_data

        if not data['episode'] or data['episode'][-1] < episode:
            data['episode'].append(episode)
            data['total_reward'].append(values['total_reward'])
            data['steps'].append(values['steps'])
        elif data['episode'][-1] == episode:
            data['total_reward'][-1] = values['total_reward']
            data['steps'][-1] = values['steps']
        else:
            print(f'skipped update for episode {episode}')

        fig = plotly.subplots.make_subplots(rows=4, cols=1, vertical_spacing=0.2)
        fig.append_trace({
            'x': eval_data['episode'],
            'y': eval_data['total_reward'],
            'name': 'Total Reward (eval)',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
        fig.append_trace({
            'x': eval_data['episode'],
            'y': eval_data['steps'],
            'name': 'Steps (eval)',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 2, 1)

        fig.append_trace({
            'x': train_data['episode'],
            'y': train_data['total_reward'],
            'name': 'Total Reward (train)',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 3, 1)
        fig.append_trace({
            'x': train_data['episode'],
            'y': train_data['steps'],
            'name': 'Steps (train)',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 4, 1)

        return fig

    app.run_server(debug=True, use_reloader=False)


Transition = namedtuple(
    'Transition',
    ['obs', 'action', 'new_obs', 'reward', 'terminated', 'truncated']
)


def train(agent, env, episodes, seed):
    for episode_idx in range(episodes):
        episode = episode_idx + 1
        obs, info = env.reset(seed=seed)
        policy = agent.explore(episode)

        terminated, truncated = False, False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = policy(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            trans = Transition(obs, action, new_obs, reward, terminated, truncated)

            # inform agent about the reward
            agent.receive(episode, trans)
            obs = new_obs
            total_reward += reward
            steps += 1

        yield (
            episode,
            {
                'total_reward': total_reward,
                'steps': steps,
                'policy': policy.stats()
            },
        )


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


def constant_episode_fn(value):
    def fn(episode):
        return value
    return fn


def _ensure_float_callable(name, value, min_value, max_value):
    if callable(value):
        return value

    if isinstance(value, float):
        if not (min_value < value <= max_value):
            raise ValueError(
                f'{name} needs to be in ({min_value}, {max_value}]'
            )
        return constant_episode_fn(value)

    raise ValueError(f'{name} neither float nor callable')


class EpsilonGreedyPolicy:

    def __init__(self, agent, epsilon):
        self.agent = agent
        self.epsilon = epsilon

    def __call__(self, observation):
        p = self.agent.rand.uniform(0, 1)
        if p < self.epsilon:
            return self.agent.action_space.sample()
        return self.agent.action(observation)

    def stats(self):
        return {'epsilon': self.epsilon}


class QLAgent:
    """ Q-Learning agent."""

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
        self.discount = discount
        self.learning_rate = learning_rate
        self.greedyness = _ensure_float_callable('greedyness', greedyness, 0, 1)

        self.rand = np.random.default_rng(seed)

        # agents state
        # self.qtable = np.zeros((observation_space.n, action_space.n))
        self.qtable = self.rand.uniform(0, 1, (observation_space.n, action_space.n))

    def action(self, observation):
        return np.argmax(self.qtable[observation, :])

    def explore(self, episode):
        return EpsilonGreedyPolicy(self, self.greedyness(episode))

    def receive(self, episode, transition):
        if transition.terminated:
            target = transition.reward
        else:
            target = (
                transition.reward
                + self.discount * np.max(self.qtable[transition.new_obs, :])
            )

        q_value = self.qtable[transition.obs, transition.action]

        self.qtable[transition.obs, transition.action] = (
            (1 - self.learning_rate) * q_value + self.learning_rate * target
        )


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
            learning_rate=0.7,
            discount=0.95,
            greedyness=0.4,
            action_space=env.action_space,
            observation_space=env.observation_space
        )

        for episode, values in train(agent, env, 10000, seed=1889):
            msg_queue.put((episode+1, 'train', values))
            if episode % 50 == 0:
                # mean steps
                for eval_episode, eval_values in episode_mean(evaluate(agent, env, 20)):
                    msg_queue.put((episode+1, 'eval', eval_values))
                # msg_queue.put(episode+1, {'policy': })

    finally:
        env.close()

    input()
