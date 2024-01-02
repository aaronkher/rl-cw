import collections
from dataclasses import dataclass
import numpy as np

import torch
from data_helper import LivePlot
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from environment import Action, PendulumEnv, State
from replay_buffer import TransitionBatch, TransitionBuffer


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
    ):
        self.episode_count = episode_count
        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size
        self.target_network_learning_rate = target_network_learning_rate

        self.environment = PendulumEnv()
        self.transition_buffer = TransitionBuffer(omega=0.5)

        self.critic_network = CriticNetwork(self.environment)
        self.target_critic_network = self.critic_network.create_copy()

        self.actor_network = ActorNetwork(self.environment, self.critic_network)
        self.target_actor_network = self.actor_network.create_copy()

    def get_action(self, state: State) -> Action:
        action = self.actor_network.get_action(state)
        self.add_ou_noise(action)
        return action

    def add_ou_noise(self, x, mu=0.0, theta=0.15, sigma=0.5):  # mu is mean, theta is friction, sigma is noise
        dx = theta * (mu - x) + sigma * np.random.randn()
        return x + dx

    def compute_td_targets(self, experiences: TransitionBatch) -> TdTargetBatch:
        # td target is:
        # reward + discounted qvalue  (if not terminal)
        # reward + 0                  (if terminal)

        new_states = experiences.new_states

        # ask the actor network what actions it would choose
        actions = self.actor_network.get_actions(new_states)

        # ask the critic network to criticize these actions
        # Tensor[[QValue * 3], [QValue * 3], ...]
        discounted_qvalues = self.critic_network.get_q_values(new_states, actions)
        discounted_qvalues[experiences.terminal] = 0
        discounted_qvalues *= self.gamma

        # Tensor[[Reward], [Reward], ...]
        rewards = experiences.rewards.unsqueeze(1)

        # Tensor[TDTarget, TDTarget, ...]
        td_targets = rewards + discounted_qvalues
        return TdTargetBatch(td_targets)

    def train_critic_network(self, experiences: TransitionBatch, td_targets: TdTargetBatch):
        self.critic_network.train(experiences, td_targets)

    def train_actor_network(self, experiences: TransitionBatch):
        self.actor_network.train(experiences)

    def update_target_networks(self):
        self.target_critic_network.polyak_update(self.critic_network, self.target_network_learning_rate)
        self.target_actor_network.polyak_update(self.actor_network, self.target_network_learning_rate)

    def decay_epsilon(self, episode: int):  # todo rename
        pass

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
                    self.transition_buffer.add(transition)

                    if self.transition_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.transition_buffer.get_batch(self.buffer_batch_size)
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
                    f" | reward {reward_sum: <7.2f} | avg {running_avg: <6.2f} (last {len(recent_rewards)})"
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
