import random

import numpy as np

from environment import Environment, Action, State, ActionTaken
from network import NeuralNetwork

class DQN:

    def __init__(
            self,
            episode_count: int,
            timestep_count: int,
            epsilon: float,
            gamma: float
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
            action = random.choice(self.environment.action_list())
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> ActionTaken:
        return self.environment.take_action(action)

    def get_q_value(self, state: State):
        action = self.get_best_action(state)
        # get logits (q values) from neural network 
        q_value = self.neural_network(state)[action]

        

        return q_value

    def compute_td_target(self, terminal_status: bool, action: Action):
        last_reward = self.environment.last_reward()
        current_state = self.environment.current_state()
        if terminal_status:
            td_target = last_reward
        else:
            td_target = last_reward + self.gamma * self.get_q_value(current_state)
        
    def gradient_descent(self):
        self.neural_network.gradient_descent()

    def train_dqn(self):
        for episode in range(self.episode_count):
            self.environment.reset()
            
            for timestep in range(self.timestep_count):              
                action = self.get_action()
                self.execute_action(action)
                
                yt = self.compute_td_target()

                self.gradient_descent()