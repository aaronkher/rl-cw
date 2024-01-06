import random
from dataclasses import dataclass
import collections
from datetime import datetime
import math
from typing import Iterable, Deque

import torch
import numpy as np
from dqn.dqn_network import DqnNetwork, DqnNetworkResult

from environment.environment import DiscreteAction, DiscreteActionEnv, Environment, State, Transition
from network import NeuralNetwork
from data_helper import LivePlot
from replay_buffer import TransitionBatch, TransitionBuffer


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class DQN:
    def __init__(
        self,
        environment: DiscreteActionEnv,
        episode_count: int,
        timestep_count: int,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        buffer_batch_size: int = 100,
        checkpoint_id: str | None = None,
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay
        self.epsilon = epsilon_start

        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size

        self.environment = environment

        self.transition_buffer = TransitionBuffer(omega=0.5)
        self.policy_network = DqnNetwork(self.environment)  # q1 / θ
        self.target_network = self.policy_network.create_copy()  # q2 / θ-

        self.latest_checkpoint = None
        self.checkpoint_id = checkpoint_id
        if checkpoint_id is not None:
            self.policy_network.load_checkpoint(checkpoint_id)
            self.target_network.load_checkpoint(checkpoint_id)

    def get_best_action(self, state: State) -> DiscreteAction:
        return self.policy_network.get_best_action(state)

    def get_action_probability_distribution(self, q_values: DqnNetworkResult) -> np.ndarray:
        # q_values: DqnNetworkResult
        # q_values.tensor: Tensor[QValue, QValue, ...]
        q_value_sum = torch.sum(q_values.tensor)
        return (q_values.tensor / q_value_sum).numpy()

    def get_action_from_probability_distribution(
        self, probability_distribution: np.ndarray
    ) -> DiscreteAction:
        # probability_distribution: np.ndarray
        # probability_distribution: [0.1, 0.2, 0.7]
        possible_actions = [a.action for a in self.environment.action_list]
        action = np.random.choice(possible_actions, p=probability_distribution)
        return DiscreteAction(action)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = self.environment.random_action()
        else:
            # pick best action
            action = self.get_best_action(state)
            # return action
        return action

    def execute_action(self, action: DiscreteAction) -> Transition:
        return self.environment.take_action(action)

    # using policy
    def get_q_values(self, state: State) -> DqnNetworkResult:
        return self.policy_network.get_q_values(state)

    # using target network here to estimate q values
    def get_q_value_for_action(self, state: State, action: DiscreteAction, policy_net=False) -> float:
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

    def compute_td_targets_batch(self, experiences: TransitionBatch) -> TdTargetBatch:
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

    def update_experiences_td_errors(self, experiences: TransitionBatch):
        td_targets = self.compute_td_targets_batch(experiences).tensor

        q_values = self.policy_network.get_q_values_batch(experiences.old_states)
        q_values = q_values.for_actions(experiences.actions).squeeze(1)

        c = 0.0001  # small constant (ϵ in Prioritized Replay Experience paper)
        td_errors = ((td_targets - q_values).abs() + c) ** self.transition_buffer.omega
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

    def backprop(self, experiences: TransitionBatch, td_targets: TdTargetBatch) -> float:
        return self.policy_network.train(experiences, td_targets)

    def train(self):
        plot = LivePlot()

        self.high_score = 0.0
        high_score_episode = 0

        start = datetime.now()

        try:
            recent_rewards = collections.deque(maxlen=30)
            for episode in range(self.episode_count):
                self.environment.reset()

                reward_sum = 0
                transition = None
                timestep = 0

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t
                    # action_probabilities = self.get_action_probability_distribution(self.get_q_values(state))
                    # action_from_policy = self.get_action_from_probability_distribution(action_probabilities)
                    action = self.get_action_using_epsilon_greedy(state)  # A_t

                    transition = self.execute_action(action)
                    reward_sum += transition.reward
                    self.transition_buffer.add(transition)

                    if self.transition_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.transition_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets_batch(replay_batch)

                        loss = self.backprop(replay_batch, td_targets)
                        plot.add_losses(loss, can_redraw=False)

                        self.update_experiences_td_errors(replay_batch)

                    self.update_target_network()

                    # process termination
                    if transition.end_of_episode():
                        break

                # episode ended
                recent_rewards.append(reward_sum)

                # print episode result
                assert transition is not None
                won = self.environment.won(transition)
                won_str = "(won) " if won else "(lost)"
                running_avg = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"Episode {episode+1: <3} | {timestep+1: >3} timesteps {won_str}"
                    f" | reward {reward_sum: <6.2f} | avg {running_avg: <6.2f} {f'(last {len(recent_rewards)})': <9}"
                    f" | ε {self.epsilon:.2f} | last_reward {transition.reward:.2f} won {won}"
                )

                now = datetime.now()
                running_for = now - start

                suffix = None  # if should create checkpoint, will be a str
                if episode == self.episode_count - 1:  # create checkpoint after 10 episodes for startup
                    suffix = " (startup checkpoint)"
                if episode % 100 == 0 and episode != 0:  # create checkpoint every 100 episodes
                    suffix = f" (ep {episode})"
                if reward_sum > self.high_score + 0.01:
                    self.high_score = reward_sum
                    if (
                        episode > high_score_episode + 15
                    ):  # create checkpoint if high score has been beaten for 15 episodes
                        high_score_episode = episode
                        suffix = f" (hs {self.high_score:.1f})"

                should_save_checkpoint = suffix is not None
                if should_save_checkpoint:
                    checkpoint_folder, self.latest_checkpoint = self.policy_network.save_checkpoint(
                        episode_number=episode,
                        reward=reward_sum,
                        won=won,
                        epsilon=self.epsilon,
                        running_since=start,
                        running_for=running_for,
                        start_checkpoint=self.checkpoint_id,
                        previous_checkpoint=self.latest_checkpoint,
                        suffix=suffix,
                    )

                    plot.save_img(checkpoint_folder / "plot.png")
                    plot.save_csv(checkpoint_folder / "data.csv")

                self.decay_epsilon(episode)

                plot.add_episode(reward_sum, won, running_avg)
                if episode % 5 == 0:
                    plot.draw()

        except KeyboardInterrupt:
            pass

        try:
            plot.draw()
        except KeyboardInterrupt:
            pass  # ctrl-c to close plot

        plot.close()
