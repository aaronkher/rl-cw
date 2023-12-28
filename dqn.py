import random
from dataclasses import dataclass
import math
import collections
import typing

import torch

from environment import Environment, Action, State, ActionResult
from network import NeuralNetwork, NeuralNetworkResult, NeuralNetworkResultBatch
from data_helper import plot_episode_data, EpisodeData


@dataclass
class Experience:
    old_state: State
    new_state: State
    action: Action
    reward: float
    terminal: bool


@dataclass
class TdTargetBatch:
    # Tensor[[TDTarget], [TDTarget], ...]
    tensor: torch.Tensor


Transition = collections.namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)

TransitionBatch = list[Transition]


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory: typing.Deque[Transition] = collections.deque([], maxlen=capacity)

    def push(
        self,
        state: State,
        action: Action,
        next_state: State | None,
        reward: torch.Tensor,
    ):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(
        self,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        buffer_batch_size: int = 100,
    ):

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay

        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size

        self.environment = Environment()

        self.replay_buffer = ReplayMemory(10000)
        self.policy_network = NeuralNetwork(self.environment)  # q1
        self.target_network = NeuralNetwork(self.environment)  # q2
        self.target_network.copy_from_other(self.policy_network)

        self.steps_taken = 0

    def get_best_action(self, state: State) -> Action:
        return self.policy_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            math.exp(-1.0 * self.steps_taken / self.decay_rate)
        )
        self.steps_taken += 1

        if random.random() < epsilon:
            # pick random action
            action_int = self.environment.env.action_space.sample()
            action = NeuralNetwork.tensorify([[action_int]])
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

    def update_target_network(self):
        # policy_network_weights = self.policy_network.state_dict()
        # self.target_network.load_state_dict(policy_network_weights)

        # tmp: update network using weighted average of both networks
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.policy_network.state_dict()
        tau = 0.05

        for key in target_net_state:
            target_net_state[key] = (
                tau * policy_net_state[key] + (1 - tau) * target_net_state[key]
            )

        self.target_network.load_state_dict(target_net_state)

    def backprop(self, transitions: TransitionBatch):
        self.policy_network.backprop(transitions, self)

    def train(self):
        episodes = []

        try:
            for episode in range(10*1000):
                print(f"Episode: {episode}")
                self.environment.reset()

                timestep = 0
                reward_sum = 0
                won = False

                for timestep in range(10*1000):
                    state = self.environment.current_state  # S_t
                    assert state is not None

                    action = self.get_action_using_epsilon_greedy(state)  # A_t
                    action_result = self.execute_action(action)
                    reward_sum += action_result.reward.item()

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    if action_result.terminal and action_result.won:
                        next_state = None
                    else:
                        next_state = action_result.new_state

                    self.replay_buffer.push(
                        action_result.old_state,
                        action_result.action,
                        next_state,
                        action_result.reward,
                    )

                    if len(self.replay_buffer) > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.sample(self.buffer_batch_size)
                        self.backprop(replay_batch)

                    self.update_target_network()

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
                # print(f"Episode {episode} finished with total reward {reward_sum}")

        except KeyboardInterrupt:
            pass

        plot_episode_data(episodes)
