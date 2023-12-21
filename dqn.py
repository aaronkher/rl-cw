import random

import numpy as np

from environment import Environment, Action, State, ActionTaken
from network import NeuralNetwork

class DQN:

    def __init__(self, episode_count: int, epsilon: float):
        self.episode_count = episode_count
        self.epsilon = epsilon

        self.environment = Environment()
        self.neural_network = NeuralNetwork()

    def get_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list())
        else:
            # pick best action
            action = self.neural_network.best_action(self.environment.current_state)
        return action

    def execute_action(self, action: Action) -> ActionTaken:
        return self.environment.take_action(action)


    def train_dqn(self):
        for episode in range(self.episodes):
            action = self.get_action()

