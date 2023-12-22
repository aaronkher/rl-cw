import random
from dataclasses import dataclass

import numpy as np

from environment import Environment, Action, State, ActionTaken
from network import NeuralNetwork, NeuralNetworkResult


@dataclass
class Experience:
    old_state: State
    new_state: State
    action: Action
    reward: float


class ReplayBuffer:
    def __init__(self):
        self.buffer: list[Experience] = []

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        return len(self.buffer)


class DQN:
    def __init__(
        self, episode_count: int, timestep_count: int, epsilon: float, gamma: float
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count
        self.epsilon = epsilon
        self.gamma = gamma

        self.environment = Environment()
        self.neural_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        self.replay_buffer = ReplayBuffer()

    def get_best_action(self, state: State) -> Action:
        return self.neural_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list)
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> ActionTaken:
        return self.environment.take_action(action)

    def get_q_values(self, state: State) -> NeuralNetworkResult:
        return self.neural_network.get_q_values(state)

    def get_q_value_for_action(self, state: State, action: Action) -> float:
        neural_network_result = self.neural_network.get_q_values(state)
        return neural_network_result.q_value_for_action(action)

    def compute_td_target(self, experience: Experience) -> float:
        # TD Target is the last reward + the expected reward of the
        # best action in the next state, discounted.

        # the reward and state after the last action was taken:
        last_reward = experience.reward  # R_t
        current_state = experience.new_state  # S_t+1

        if self.environment.is_terminated:
            td_target = last_reward
        else:
            action = self.get_best_action(current_state)
            td_target = last_reward + self.gamma * self.get_q_value_for_action(
                current_state, action
            )

        return td_target

    def backprop(self, nn_result: NeuralNetworkResult, td_target: float):
        self.neural_network.backprop(nn_result, td_target)

    def train(self):
        for episode in range(self.episode_count):
            print(f"Episode: {episode}")
            self.environment.reset()

            reward_sum = 0

            for timestep in range(self.timestep_count):
                if timestep % 100 == 0:
                    print(f"Timestep: {timestep}")
                    self.environment.render()

                state = self.environment.current_state  # S_t

                action = self.get_action_using_epsilon_greedy(state)  # A_t
                action_taken = self.execute_action(action)

                experience = Experience(
                    action_taken.old_state,
                    action_taken.new_state,
                    action,
                    action_taken.reward,
                )
                self.replay_buffer.add_experience(experience)
                
                if self.replay_buffer.size() <= 20:
                    continue

                replay_batch = self.replay_buffer.get_batch(20)
                for replay in replay_batch:
                    y_t = self.compute_td_target(replay)
                    y_hat = self.get_q_values(state)

                    self.backprop(y_hat, y_t)

            print(f"Episode {episode} finished with total reward {reward_sum}")
