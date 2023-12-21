import random

import numpy as np

from environment import Environment, Action, State, ActionTaken
from network import NeuralNetwork


class DQN:
    def __init__(
        self, episode_count: int, timestep_count: int, epsilon: float, gamma: float
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count
        self.epsilon = epsilon
        self.gamma = gamma

        self.environment = Environment()
        self.neural_network = NeuralNetwork()

    def get_best_action(self, state: State) -> Action:
        return self.neural_network.best_action(state)

    def get_action(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list)
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> ActionTaken:
        return self.environment.take_action(action)

    def get_q_value_for_action(self, state: State, action: Action) -> float:
        q_values = self.neural_network(
            state
        )  # get logits (q values) from neural network (also calls forward method)
        q_value_for_action = q_values[action]
        return q_value_for_action

    def compute_td_target(self):
        last_reward = self.environment.last_reward
        current_state = (
            self.environment.current_state
        )  # this represents the state after the action is taken (S_t+1)

        if self.environment.is_terminated:
            td_target = last_reward
        else:
            action = self.get_best_action(current_state)
            td_target = last_reward + self.gamma * self.get_q_value_for_action(
                current_state, action
            )

        return td_target

    def gradient_descent(self, prediction: float, label: float):
        self.neural_network.gradient_descent(prediction, label)

    def train_dqn(self):
        for episode in range(self.episode_count):
            self.environment.reset()

            for timestep in range(self.timestep_count):
                state = self.environment.current_state  # S_t

                action = self.get_action(state)  # A_t
                self.execute_action(action)

                y_t = self.compute_td_target()
                y_hat = self.get_q_value_for_action(state, action)

                self.gradient_descent(y_t, y_hat)
