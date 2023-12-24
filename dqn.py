import random
from dataclasses import dataclass
import torch

import numpy as np
import math

from environment import Environment, Action, State, ActionResult
from network import NeuralNetwork, NeuralNetworkResult, NeuralNetworkResultBatch
from data_helper import plot_episode_data, EpisodeData


@dataclass
class Experience:
    old_state: State
    new_state: State
    action: Action
    reward: float


class ExperienceBatch:
    def __init__(self, experiences: list[Experience]):
        self.size = len(experiences)

        # tensorify = lambda prop_name: NeuralNetwork.tensorify(
        #     [getattr(exp, prop_name) for exp in experiences]
        # )
        # tensorify = lambda prop_name: NeuralNetwork.tensorify(
        #     [getattr(exp, prop_name) for exp in experiences]
        # )
        # self.actions = tensorify("action")
        # self.rewards = tensorify("reward")

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([[exp.action] for exp in experiences])

        # Tensor[-0.99, -0.99, ...]
        self.rewards = NeuralNetwork.tensorify([exp.reward for exp in experiences])

        # Tensor[State, State, ...]
        # states are already torch tensors, so we can just use torch.stack
        self.old_states = torch.stack([exp.old_state for exp in experiences])


class ReplayBuffer:
    def __init__(self):
        self.buffer: list[Experience] = []
        self.max_buffer_len = 10000

    def add_experience(self, experience: Experience):
        if len(self.buffer) >= self.max_buffer_len:
            self.buffer.pop(0)  # remove oldest
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> ExperienceBatch:
        experiences = random.sample(self.buffer, batch_size)
        return ExperienceBatch(experiences)

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
        self.buffer_batch_size = 100

        self.environment = Environment()

        # initialise replay memory
        self.replay_buffer = ReplayBuffer()
        # initialise q1
        self.policy_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # initialise q2
        self.target_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # copy q2 to q1
        self.policy_network.load_state_dict(self.target_network.state_dict())

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

    def compute_td_targets_batch(
        self, experiences: ExperienceBatch
    ) -> NeuralNetworkResultBatch:
        return self.policy_network.get_q_values_batch(experiences.old_states)

    def decay_epsilon(self, episode):
        # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
        self.epsilon = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-self.decay_rate * episode)

    def update_target_network(self):
        policy_network_weights = self.policy_network.state_dict()
        self.target_network.load_state_dict(policy_network_weights)

    def backprop(
        self, experiences: ExperienceBatch, td_targets: NeuralNetworkResultBatch
    ):
        self.policy_network.backprop(experiences, td_targets)

    def train(self):
        episodes = []

        try:
            timestep_C_count = 0
            for episode in range(self.episode_count):
                print(f"Episode: {episode}")
                self.environment.reset()

                timestep = 0
                reward_sum = 0
                won = False

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t

                    action = self.get_action_using_epsilon_greedy(state)  # A_t
                    action_result = self.execute_action(action)

                    action_result = self.execute_action(action)
                    reward_sum += action_result.reward

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    experience = Experience(
                        action_result.old_state,
                        action_result.new_state,
                        action,
                        action_result.reward,
                    )
                    self.replay_buffer.add_experience(experience)

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(
                            self.buffer_batch_size
                        )
                        td_targets = self.compute_td_targets_batch(replay_batch)

                        self.backprop(replay_batch, td_targets)

                    timestep_C_count += 1
                    if timestep_C_count == self.C:
                        self.update_target_network()
                        timestep_C_count = 0

                    # process termination
                    if action_result.terminal:
                        won = action_result.won
                        won_str = "won" if won else "lost"
                        print(
                            f"Episode {episode+1} ended ({won_str}) after {timestep+1} timestaps"
                            f" with total reward {reward_sum:.2f}"
                        )
                        break

                episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                self.decay_epsilon(episode)
                # print(f"Episode {episode} finished with total reward {reward_sum}")

        except KeyboardInterrupt:
            pass

        plot_episode_data(episodes)
