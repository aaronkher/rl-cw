import collections
from dataclasses import dataclass

import torch
import numpy as np
from data_helper import LivePlot
from environment import Action, MountainCarContinuousEnv, State
from replay_buffer import TransitionBatch, ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork

@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class DDPG:
    def __init__(
        self,
        episode_count: int,
        gamma: float,
        buffer_batch_size: int,
        target_network_learning_rate: float,
        sigma: float = 0.15,
    ):
        self.episode_count = episode_count
        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size
        self.target_network_learning_rate = target_network_learning_rate
        self.sigma = sigma

        self.environment = MountainCarContinuousEnv()
        self.replay_buffer = ReplayBuffer()

        # {} used as placeholders
        self.actor_network = ActorNetwork(self.environment)
        self.target_actor_network = self.actor_network.create_copy()
        # copy weights

        self.critic_network = CriticNetwork(self.environment)
        self.target_critic_network = self.critic_network.create_copy()
        # copy weights

    def compute_OU_noise(self, mu: float, theta: float, sigma: float) -> float:
        # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        return theta * (mu - sigma) + sigma * np.random.randn(1)

    def get_action(self, state: State) -> Action:
        perfect_action = self.actor_network.get_action(state)
        noise = self.compute_OU_noise(0, 0.15, self.sigma)
        action = perfect_action + noise
        return action

    def compute_td_targets(self, experiences: TransitionBatch) -> TdTargetBatch:
        # td target is:
        # reward + discounted qvalue  (if not terminal)
        # reward + 0                  (if terminal)

        # Tensor[-0.99, -0.99, ...]
        rewards = experiences.rewards

        next_actions = self.target_actor_network.get_action_batch(experiences.new_states)

        # Tensor[[QValue], [QValue], ...]
        qvalues = self.target_critic_network.get_q_values(
            experiences.new_states,
            next_actions
        )

        discounted_qvalues_tensor = qvalues * self.gamma
        discounted_qvalues_tensor[experiences.terminal] = 0

        # # reformat rewards tensor to same shape as discounted_qvalues_tensor
        # # Tensor[[-0.99], [-0.99], ...]
        rewards = rewards.unsqueeze(1)

        # Tensor[[TDTarget], [TDTarget], ...]
        td_targets = rewards + discounted_qvalues_tensor

        return TdTargetBatch(td_targets)

    def train_critic_network(self, experiences: TransitionBatch, td_targets: TdTargetBatch):
        self.critic_network.train(experiences, td_targets)

    def train_actor_network(self, experiences: TransitionBatch):
        self.actor_network.train(experiences, self.critic_network)

    def update_target_networks(self):
        self.target_actor_network.update_weights(self.actor_network, self.target_network_learning_rate)
        self.target_critic_network.update_weights(self.critic_network, self.target_network_learning_rate)

    def decay_noise(self, episode: int):  # todo rename
        self.sigma = max(0.01, self.sigma - 0.0001)

    def train(self):
        plot = LivePlot()
        plot.create_figure()

        try:
            recent_rewards = collections.deque(maxlen=30)
            for episode in range(self.episode_count):
                self.environment.reset()

                reward_sum = 0
                transition = None
                timestep = 0

                for timestep in range(10 * 1000):
                    state = self.environment.current_state  # S_t
                    action = self.get_action(state)  # A_t

                    transition = self.environment.take_action(action)
                    reward_sum += transition.reward
                    self.replay_buffer.add_experience(transition)

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets(replay_batch)

                        self.train_critic_network(replay_batch, td_targets)
                        self.train_actor_network(replay_batch)

                    self.update_target_networks()

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
                )

                self.decay_epsilon(episode)

                # episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                plot.add_episode(reward_sum, won, running_avg)
                if episode % 5 == 0:
                    plot.draw()

        except KeyboardInterrupt:  # ctrl-c received while training
            pass  # stop training

        try:
            plot.draw()  # draw final results
        except KeyboardInterrupt:  # ctrl-c used to close final results
            pass  # we try/except/pass to not print KeyboardInterrupt error
