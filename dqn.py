import random
from dataclasses import dataclass
import torch

import numpy as np
import math

from environment import Environment, Action, State, ActionResult
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
        self.epsilon_min = 0.01
        self.epsilon_start = 0.9
        self.decay_rate = 0.05

        self.gamma = gamma
        self.C = 50  # TODO: don't harcode this
        self.buffer_batch_size = 50

        self.environment = Environment()

        # initialise replay memory
        self.replay_buffer = ReplayBuffer()
        # initialise q1
        self.policy_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # initialise q2
        self.target_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())

    def get_best_action(self, state: State) -> Action:
        return self.policy_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list)
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> ActionResult:
        return self.environment.take_action(action)

    # using policy
    def get_q_values(self, state: State) -> NeuralNetworkResult:
        return self.policy_network.get_q_values(state)

    # using target network here to estimate q values
    def get_q_value_for_action(self, state: State, action: Action) -> float:
        neural_network_result = self.target_network.get_q_values(state)
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

    def decay_epsilon(self, episode):
        # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
        self.epsilon = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-self.decay_rate * episode)

    def update_target_network(self):
        policy_network_weights = self.policy_network.state_dict()
        self.target_network.load_state_dict(policy_network_weights)

    def backprop(self, nn_result: NeuralNetworkResult, td_target: float):
        self.policy_network.backprop(nn_result, td_target)

    def train(self):
        timestep_C_count = 0
        for episode in range(self.episode_count):
            print(f"Episode: {episode}")
            self.environment.reset()

            reward_sum = 0

            for timestep in range(self.timestep_count):
                if timestep % 100 == 0:
                    print(f"Timestep: {timestep}, total reward {reward_sum}")
                    # self.environment.render()

                state = self.environment.current_state  # S_t

                action = self.get_action_using_epsilon_greedy(state)  # A_t
                action_result = self.execute_action(action)

                action_result = self.execute_action(action)
                reward_sum += action_result.reward

                experience = Experience(
                    action_result.old_state,
                    action_result.new_state,
                    action,
                    action_result.reward,
                )
                self.replay_buffer.add_experience(experience)

                if self.replay_buffer.size() <= self.buffer_batch_size:
                    continue

                replay_batch = self.replay_buffer.get_batch(self.buffer_batch_size)
                for replay in replay_batch:
                    y_t = self.compute_td_target(replay)
                    y_hat = self.get_q_values(state)

                    self.backprop(y_hat, y_t)

                timestep_C_count += 1
                if timestep_C_count == self.C:
                    self.update_target_network()
                    timestep_C_count = 0

                # process termination
                if action_result.terminated:
                    print(
                        f"Episode {episode} terminated with total reward {reward_sum}"
                    )
                    break

                # process termination
                if action_result.terminated:
                    print(
                        f"Episode {episode} terminated with total reward {reward_sum}"
                    )
                    break

                if action_result.truncated:
                    print(f"Episode {episode} truncated with total reward {reward_sum}")
                    break
            self.decay_epsilon(episode)
            # print(f"Episode {episode} finished with total reward {reward_sum}")
