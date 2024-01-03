import random
from dataclasses import dataclass
import math
import collections
from typing import Iterable, Deque

import torch
import numpy as np
from trackmania_dqn.dqn_network import TrackManiaDqnNetwork, DqnNetworkResult

from environment import CartpoleEnv, Environment, Action, State, Transition
from network import NeuralNetwork
from data_helper import LivePlot
from replay_buffer import TransitionBatch, TransitionBuffer


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class TrackManiaDQN:
    def __init__(
        self,
        observation_space,
        action_space,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        omega: float = 0.5,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay
        self.epsilon = epsilon_start
        self.omega = omega

        self.gamma = gamma

        self.observation_space = observation_space
        self.action_space = action_space

        self.policy_network = TrackManiaDqnNetwork(observation_space, action_space)  # q1 / θ
        self.target_network = self.policy_network.create_copy()  # q2 / θ-

    def get_best_action(self, state: State) -> Action:
        return self.policy_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(list(range(self.action_space.shape[0]))) # TODO select random action - figure out discrete action space
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    # using policy
    def get_q_values(self, state: State) -> DqnNetworkResult:
        return self.policy_network.get_q_values(state)

    def compute_td_targets_batch(self, next_states, rewards, terminal) -> TdTargetBatch:
        # using double dqn:
        # td_target = R_t+1 + γ * max_a' q_θ-(S_t+1, argmax_a' q_θ(S_t+1, a'))
        # the best action in S_t+1, according to the policy network
        best_actions = self.policy_network.get_q_values_batch(next_states).best_actions()
        best_actions = best_actions.unsqueeze(1)

        # the q-value of that action, according to the target network
        q_values = self.target_network.get_q_values_batch(next_states).for_actions(best_actions)
        q_values = q_values.squeeze(1)
        # TODO how to handle terminal states - TMRL docs suggest ignoring truncated but not terminal
        q_values[terminal] = 0
        q_values *= self.gamma

        # Tensor[TDTarget, TDTarget, ...]
        td_targets = rewards + q_values
        return TdTargetBatch(td_targets)

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

    def backprop(self, prev_states, actions_chosen, td_targets: TdTargetBatch):
        self.policy_network.train(prev_states, actions_chosen, td_targets)

    def train(self, batch):
        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        td_targets = self.compute_td_targets_batch(o2, r, d)

        self.backprop(o, a, td_targets)
        # self.update_experiences_td_errors(batch) # for prioritized replay so we can comment

        self.update_target_network()