import random
from dataclasses import dataclass
import math
import collections
from typing import Iterable, Deque

import torch
import numpy as np
from dqn.dqn_network import DqnNetwork, DqnNetworkResult

from environment import CartpoleEnv, Environment, Action, State, Transition
from network import NeuralNetwork
from data_helper import LivePlot
from replay_buffer import ExperienceBatch, ReplayBuffer


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class DQN:
    def __init__(
        self,
        episode_count: int,
        timestep_count: int,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        buffer_batch_size: int = 100,
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay
        self.epsilon = epsilon_start

        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size

        self.environment = CartpoleEnv()

        self.replay_buffer = ReplayBuffer()
        self.policy_network = DqnNetwork(self.environment)  # q1 / θ
        self.target_network = self.policy_network.create_copy()  # q2 / θ-

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

    def execute_action(self, action: Action) -> Transition:
        return self.environment.take_action(action)

    # using policy
    def get_q_values(self, state: State) -> DqnNetworkResult:
        return self.policy_network.get_q_values(state)

    # using target network here to estimate q values
    def get_q_value_for_action(self, state: State, action: Action, policy_net=False) -> float:
        network = self.policy_network if policy_net else self.target_network
        neural_network_result = network.get_q_values(state)
        return neural_network_result.q_value_for_action(action)

    def compute_td_target(self, experience: Transition) -> float:
        # TD Target is the last reward + the expected reward of the
        # best action in the next state, discounted.
        # Note: this function does not use double dqn! (yet)

        # the reward and state after the last action was taken:
        last_reward = experience.reward  # R_t
        current_state = experience.new_state  # S_t+1

        if current_state.terminal:  # terminal experience
            td_target = last_reward
        else:
            target_net_result = self.target_network.get_q_values(current_state)
            best_q_value = target_net_result.best_action_q_value()
            td_target = last_reward + self.gamma * best_q_value

        return td_target

    def compute_td_targets_batch(self, experiences: ExperienceBatch) -> TdTargetBatch:
        # using double dqn:
        # td_target = R_t+1 + γ * max_a' q_θ-(S_t+1, argmax_a' q_θ(S_t+1, a'))

        # the best action in S_t+1, according to the policy network
        best_actions = self.policy_network.get_q_values_batch(experiences.new_states).best_actions()
        best_actions = best_actions.unsqueeze(1)

        # the q-value of that action, according to the target network
        q_values = self.target_network.get_q_values_batch(experiences.new_states).for_actions(best_actions)
        q_values = q_values.squeeze(1)
        q_values[experiences.terminal] = 0
        q_values *= self.gamma

        # Tensor[-0.99, -0.99, ...]
        rewards = experiences.rewards

        # Tensor[TDTarget, TDTarget, ...]
        td_targets = rewards + q_values
        return TdTargetBatch(td_targets)

    def update_experiences_td_errors(self, experiences: ExperienceBatch):
        td_targets = self.compute_td_targets_batch(experiences).tensor

        q_values = self.policy_network.get_q_values_batch(experiences.old_states)
        q_values = q_values.for_actions(experiences.actions).squeeze(1)

        c = 0.0001  # small constant (ϵ in Prioritized Replay Experience paper)
        td_errors = ((td_targets - q_values).abs() + c) ** self.replay_buffer.omega
        td_errors = td_errors.detach().cpu().numpy()
        experiences.update_td_errors(td_errors)

    def decay_epsilon(self, episode):
        # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(
            -self.decay_rate * episode
        )

    def update_target_network(self):
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.policy_network.state_dict()
        tau = 0.005

        for key in policy_net_state:
            target_net_state[key] = tau * policy_net_state[key] + (1 - tau) * target_net_state[key]

        self.target_network.load_state_dict(target_net_state)

    def backprop(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        self.policy_network.train(experiences, td_targets)

    def train(self):
        episodes = []
        plot = LivePlot()
        plot.create_figure()

        try:
            timestep_C_count = 0
            recent_rewards = collections.deque(maxlen=30)
            for episode in range(self.episode_count):
                self.environment.reset()

                reward_sum = 0
                transition = None
                timestep = 0

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t
                    action = self.get_action_using_epsilon_greedy(state)  # A_t

                    transition = self.execute_action(action)
                    reward_sum += transition.reward
                    self.replay_buffer.add_experience(transition)

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets_batch(replay_batch)

                        self.backprop(replay_batch, td_targets)
                        self.update_experiences_td_errors(replay_batch)

                    self.update_target_network()

                    # process termination
                    if transition.end_of_episode():
                        break

                # episode ended
                recent_rewards.append(reward_sum)

                # print episode result
                assert transition is not None
                won = transition.truncated
                won_str = "(won) " if won else "(lost)"
                running_avg = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"Episode {episode+1: <3} | {timestep+1: >3} timesteps {won_str}"
                    f" | reward {reward_sum: <6.2f} | avg {running_avg: <6.2f} (last {len(recent_rewards): <2})"
                    f" | ε {self.epsilon:.2f}"
                )

                self.decay_epsilon(episode)

                # episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                plot.add_episode(reward_sum, won, running_avg)
                if episode % 5 == 0:
                    plot.draw()

        except KeyboardInterrupt:
            pass

        try:
            plot.draw()
        except KeyboardInterrupt:
            pass  # ctrl-c to close plot
